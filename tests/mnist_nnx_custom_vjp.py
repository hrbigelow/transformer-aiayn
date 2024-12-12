import optax
import fire
import tensorflow_datasets as tfds  # TFDS to download MNIST.
from flax import nnx
from flax.nnx import extract, graph
from flax.nnx.transforms import general
from flax.nnx.training import optimizer as optim
import jax
import typing as tp
import functools
from functools import partial
import tensorflow as tf  # TensorFlow / `tf.data` operations.
# from aiayn.mod_optimizer import ModuleOptimizer

def mnist_dataset(split='train', seed=0, batch_size=32, total_steps=10000):
  ds: tf.data.Dataset = tfds.load('mnist', split=split)
  tf.random.set_seed(seed)  # Set the random seed for reproducibility.
  ds = ds.map(
    lambda sample: {
      'image': tf.cast(sample['image'], tf.float32) / 255,
      'label': sample['label'],
    }
  )
  if split == 'train':
    ds = ds.repeat().shuffle(1024)

  ds = ds.batch(batch_size, drop_remainder=True)
  if split == 'train': 
    ds = ds.take(total_steps)
  ds = ds.prefetch(1)
  return ds

F = tp.TypeVar('F', bound=tp.Callable[..., tp.Any])

@graph.update_context('vjp')
def nnx_vjp(f, ins, out_g):
  g = general.merge_inputs(f, ctxtag='vjp')
  pure_ins = extract.to_tree(ins, ctxtag='vjp')
  h = lambda a: g(a)[1]            # h: a -> b
  _, vjp_fn = jax.vjp(h, pure_ins) # vjp_fn: b' -> (a,)
  ins_g, = vjp_fn(out_g)
  if isinstance(ins_g, extract.NodeStates):
    return ins_g.state
  return ins_g

class Reactive(nnx.Module):
  def __init__(self, module_cls: 
               tp.Type, 
               tx: optax.GradientTransformation,
               do_imm_grad,
               *args, **kwargs):
    self.mod = module_cls(*args, **kwargs)
    self.opt = MyOptimizer(self.mod, tx)
    self.imm_fn = make_custom_vjp(do_imm_grad)

  def __call__(self, x: jax.Array):
    return self.imm_fn(self.opt, x)

def make_custom_vjp(do_imm_grad=True):
  """
  Return a function which is enabled for immediately updated gradient 
  """
  def fn(opt, x):
    return opt.model(x)

  if not do_imm_grad:
    return fn

  @nnx.custom_vjp
  def primal(opt, x):
    return fn(opt, x)

  def fn_fwd(opt, x):
    return fn(opt, x), (opt, x)

  def fn_bwd(res, g):
    ins_g, out_g = g
    opt, x = res
    opt_g = nnx_vjp(lambda opt_arg: opt_arg.model(x), opt, out_g)
    # param_g = jax.lax.pmean(param_g, 'dev') # ? each gradient on a different device
    new_params = opt.get_updated(opt_g['model'])
    graphdef = nnx.graphdef(opt.model)
    new_model = nnx.merge(graphdef, new_params)
    data_g = nnx_vjp(lambda x_arg: new_model(x_arg), x, out_g)
    # print(f'in do_imm_grad')
    # opt.update(opt_g['model']) 
    # print(f'otherwise')
    return (opt_g, data_g) # TODO: param_g unnecessary

  primal.defvjp(fn_fwd, fn_bwd)
  return primal

class MyOptimizer(nnx.Optimizer):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def get_updated(self, grads):
    """
    Get the updated params without updating any state
    """
    params = nnx.state(self.model, self.wrt)
    opt_state = optim._opt_state_variables_to_state(self.opt_state)

    updates, new_opt_state = self.tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    assert isinstance(new_params, nnx.State)
    return new_params

class CNN(nnx.Module):
  """A simple CNN model."""

  def __init__(self, 
               tx: optax.GradientTransformation, 
               do_imm_grad: bool,
               rngs: nnx.Rngs):
    self.conv1 = Reactive(nnx.Conv, tx, do_imm_grad, 1, 32, kernel_size=(3, 3), rngs=rngs)
    self.conv2 = Reactive(nnx.Conv, tx, do_imm_grad, 32, 64, kernel_size=(3, 3), rngs=rngs)
    self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
    self.linear1 = Reactive(nnx.Linear, tx, do_imm_grad, 3136, 256, rngs=rngs)
    self.linear2 = Reactive(nnx.Linear, tx, do_imm_grad, 256, 10, rngs=rngs)

  def __call__(self, x):
    x = self.avg_pool(nnx.relu(self.conv1(x)))
    x = self.avg_pool(nnx.relu(self.conv2(x)))
    x = x.reshape(x.shape[0], -1)  # flatten
    x = nnx.relu(self.linear1(x))
    x = self.linear2(x)
    return x


def main(seed=0,
         learning_rate=0.0005, 
         momentum=0.9, 
         eval_every=100, 
         train_steps=20000,
         do_eval=False,
         do_imm_grad=True):
  # tx = optax.adamw(learning_rate, momentum)
  tx = optax.sgd(learning_rate)
  model = CNN(tx, do_imm_grad, rngs=nnx.Rngs(seed))
  optimizer = nnx.Optimizer(model, tx)

  metrics = nnx.MultiMetric(
    accuracy=nnx.metrics.Accuracy(),
    loss=nnx.metrics.Average('loss'),
  )

  def loss_fn(model: CNN, batch):
    logits = model(batch['image'])
    loss = optax.softmax_cross_entropy_with_integer_labels(
      logits=logits, labels=batch['label']
    ).mean()
    return loss, logits

  @nnx.jit
  def train_step(model: CNN, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
    """Train for a single step."""
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch['label'])  # In-place updates.
    optimizer.update(grads)  # In-place updates.

  @nnx.jit
  def eval_step(model: CNN, metrics: nnx.MultiMetric, batch):
    loss, logits = loss_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch['label'])  # In-place updates.


  metrics_history = {
    'train_loss': [],
    'train_accuracy': [],
    'test_loss': [],
    'test_accuracy': [],
  }

  batch_size = 32
  train_ds = make_dataset('train', 0, batch_size, train_steps)
  test_ds = make_dataset('test', 10, batch_size, 0)

  for step, batch in enumerate(train_ds.as_numpy_iterator()):
    # Run the optimization for one step and make a stateful update to the following:
    # - The train state's model parameters
    # - The optimizer state
    # - The training loss and accuracy batch metrics
    train_step(model, optimizer, metrics, batch)
    # print(f'{model.conv1.opt.step = }, {optimizer.step = }')

    if step > 0 and (step % eval_every == 0 or step == train_steps - 1):  # One training epoch has passed.
      # Log the training metrics.
      for metric, value in metrics.compute().items():  # Compute the metrics.
        metrics_history[f'train_{metric}'].append(value)  # Record the metrics.
      metrics.reset()  # Reset the metrics for the test set.

      # Compute the metrics on the test set after each training epoch.
      if do_eval:
        for test_batch in test_ds.as_numpy_iterator():
          eval_step(model, metrics, test_batch)

        # Log the test metrics.
        for metric, value in metrics.compute().items():
          metrics_history[f'test_{metric}'].append(value)
        metrics.reset()  # Reset the metrics for the next training epoch.

      print(
          f"imm_grad: {do_imm_grad} [train] step: {step}, "
          f"loss: {metrics_history['train_loss'][-1]:5.3f}, "
          f"accuracy: {metrics_history['train_accuracy'][-1] * 100:5.3f}"
      )
      if do_eval:
        print(
            f"imm_grad: {do_imm_grad} [test ] step: {step}, "
            f"loss: {metrics_history['test_loss'][-1]:5.3f}, "
            f"accuracy: {metrics_history['test_accuracy'][-1] * 100:5.3f}"
        )

if __name__ == '__main__':
  fire.Fire(main)

