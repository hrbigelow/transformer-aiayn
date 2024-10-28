import optax
import tensorflow_datasets as tfds  # TFDS to download MNIST.
from flax import nnx
from flax.nnx import extract, graph
from flax.nnx.transforms import general
import jax
import typing as tp
import functools
from functools import partial
import tensorflow as tf  # TensorFlow / `tf.data` operations.
from aiayn.mod_optimizer import ModuleOptimizer

tf.random.set_seed(0)  # Set the random seed for reproducibility.

train_steps = 12000
eval_every = 100
batch_size = 32

train_ds: tf.data.Dataset = tfds.load('mnist', split='train')
test_ds: tf.data.Dataset = tfds.load('mnist', split='test')

train_ds = train_ds.map(
  lambda sample: {
    'image': tf.cast(sample['image'], tf.float32) / 255,
    'label': sample['label'],
  }
)  # normalize train set
test_ds = test_ds.map(
  lambda sample: {
    'image': tf.cast(sample['image'], tf.float32) / 255,
    'label': sample['label'],
  }
)  # Normalize the test set.

# Create a shuffled dataset by allocating a buffer size of 1024 to randomly draw elements from.
train_ds = train_ds.repeat().shuffle(1024)
# Group into batches of `batch_size` and skip incomplete batches, prefetch the next sample to improve latency.
train_ds = train_ds.batch(batch_size, drop_remainder=True).take(train_steps).prefetch(1)
# Group into batches of `batch_size` and skip incomplete batches, prefetch the next sample to improve latency.
test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(1)

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

DO_CUSTOM_VJP = True
DO_IMM_GRAD = True
# DO_IMM_GRAD = True

class Reactive(nnx.Module):
  def __init__(self, module_cls: tp.Type, *args, **kwargs):
    self.mod = module_cls(*args, **kwargs)

  def post_init(self, opt: ModuleOptimizer):
    self.opt = opt
    if DO_CUSTOM_VJP:
      module_path = opt.find_node_path(self.mod)
      self.imm_fn = make_custom_vjp(lambda mod, x: mod(x), self.opt, module_path)
    else:
      self.imm_fn = lambda mod, x: mod(x)

  def __call__(self, x: jax.Array):
    return self.imm_fn(self.mod, x)

def make_custom_vjp(fn, mod_optimizer: ModuleOptimizer, module_path: tp.Tuple[str, ...]):
  """
  Return a function which is enabled for immediately updated gradient 
  """
  @nnx.custom_vjp
  def primal(mod, x):
    return fn(mod, x)

  def fn_fwd(mod, x):
    return fn(mod, x), (mod, x)

  def fn_bwd(res, g):
    ins_g, out_g = g
    m, x = res
    param_g = nnx_vjp(lambda mod: fn(mod, x), m, out_g)
    # param_g = jax.lax.pmean(param_g, 'dev') # ? each gradient on a different device
    if DO_IMM_GRAD:
      mod_optimizer.update(m, module_path, param_g) # per-module updates 
    data_g = nnx_vjp(lambda data: fn(m, data), x, out_g)
    # print(f'{param_g = }, {data_g = }')
    return (param_g, data_g) # TODO: param_g unnecessary

  primal.defvjp(fn_fwd, fn_bwd)
  return primal


class CNN(nnx.Module):
  """A simple CNN model."""

  def __init__(self, *, rngs: nnx.Rngs):
    self.conv1 = Reactive(nnx.Conv, 1, 32, kernel_size=(3, 3), rngs=rngs)
    self.conv2 = Reactive(nnx.Conv, 32, 64, kernel_size=(3, 3), rngs=rngs)
    self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
    self.linear1 = Reactive(nnx.Linear, 3136, 256, rngs=rngs)
    self.linear2 = Reactive(nnx.Linear, 256, 10, rngs=rngs)

  def __call__(self, x):
    x = self.avg_pool(nnx.relu(self.conv1(x)))
    x = self.avg_pool(nnx.relu(self.conv2(x)))
    x = x.reshape(x.shape[0], -1)  # flatten
    x = nnx.relu(self.linear1(x))
    x = self.linear2(x)
    return x

# Instantiate the model.
model = CNN(rngs=nnx.Rngs(0))
# print('\n'.join(str(p) + ' ' + m.__class__.__name__ for p, m in model.iter_modules()))

learning_rate = 0.005
momentum = 0.9
tx = optax.adamw(learning_rate, momentum)
optimizer = nnx.Optimizer(model, tx)
mod_optimizer = ModuleOptimizer(tx)  
mod_optimizer.post_init(model)

# post-init
for _, mod in model.iter_modules():
    if hasattr(mod, 'post_init'):
        mod.post_init(mod_optimizer)


for p, m in model.iter_modules():
    p2 = mod_optimizer.find_node_path(m)
    assert p == p2, f'{p = } != {p2 = }'

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
  if not DO_IMM_GRAD:
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

print(f'{DO_IMM_GRAD=}, {DO_CUSTOM_VJP=}')

for step, batch in enumerate(train_ds.as_numpy_iterator()):
  # Run the optimization for one step and make a stateful update to the following:
  # - The train state's model parameters
  # - The optimizer state
  # - The training loss and accuracy batch metrics
  train_step(model, optimizer, metrics, batch)

  if step > 0 and (step % eval_every == 0 or step == train_steps - 1):  # One training epoch has passed.
    # Log the training metrics.
    for metric, value in metrics.compute().items():  # Compute the metrics.
      metrics_history[f'train_{metric}'].append(value)  # Record the metrics.
    metrics.reset()  # Reset the metrics for the test set.

    # Compute the metrics on the test set after each training epoch.
    for test_batch in test_ds.as_numpy_iterator():
      eval_step(model, metrics, test_batch)

    # Log the test metrics.
    for metric, value in metrics.compute().items():
      metrics_history[f'test_{metric}'].append(value)
    metrics.reset()  # Reset the metrics for the next training epoch.

    print(
      f"[train] step: {step}, "
      f"loss: {metrics_history['train_loss'][-1]:5.3f}, "
      f"accuracy: {metrics_history['train_accuracy'][-1] * 100:5.3f}"
    )
    print(
      f"[test ] step: {step}, "
      f"loss: {metrics_history['test_loss'][-1]:5.3f}, "
      f"accuracy: {metrics_history['test_accuracy'][-1] * 100:5.3f}"
    )

  



