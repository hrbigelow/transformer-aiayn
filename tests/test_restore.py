import fire
import orbax.checkpoint as ocp
import jax
import jax.numpy as jnp

def main(ckpt_dir, step):
    options = ocp.CheckpointManagerOptions(save_interval_steps=1)
    checkpointer = ocp.PyTreeCheckpointer()
    manager = ocp.CheckpointManager(ckpt_dir, checkpointer, options)

    """
    tuple_data = (
            dict(a=jnp.array([1,3,5]), b=jnp.array([2,4,6])),
            dict(c=jnp.array([1,3,5]), d=jnp.array([2,4,6])))

    main_data = dict(opt_state=tuple_data, state=jnp.array([6,7,8]))
    save_args = jax.tree_map(lambda _: ocp.SaveArgs(aggregate=True), main_data)

    manager.save(step, main_data, save_kwargs={'save_args': save_args}, force=True)
    """
    # dest_dir = f'{ckpt_dir}/{step}/default'
    # checkpointer.save(dest_dir, main_data, force=True, save_args=save_args)
    restored = manager.restore(step, items=None, restore_kwargs=None)

    # restored = checkpointer.restore(dest_dir)
    print(type(restored['opt_state']))
    # print(restored['opt_state']['0'])



if __name__ == '__main__':
    fire.Fire(main)


