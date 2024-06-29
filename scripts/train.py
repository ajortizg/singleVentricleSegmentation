import hydra
import os.path as osp
from omegaconf import DictConfig
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping


@hydra.main(config_path="../conf", config_name="train", version_base=None)
def main(cfg: DictConfig):

    pl.seed_everything(42, workers=True)

    model = hydra.utils.instantiate(cfg["model"])
    data = hydra.utils.instantiate(cfg["data"])

    logger = TensorBoardLogger(
        save_dir=osp.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "tb_logs"),
        name=model.__class__.__name__
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=1000,
        min_epochs=1000,
        callbacks=[
            ModelCheckpoint(monitor="loss_val/t", save_top_k=1, verbose=True),
            EarlyStopping(monitor="loss_val/t", mode="min", patience=cfg["patience"], verbose=True)
        ],
        num_sanity_val_steps=-1,
        devices=cfg["gpu"],
        precision=32,
        logger=logger,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        enable_model_summary=True
    )

    trainer.fit(model, data)

    # best_ckpt = Path(checkpoint_cb.best_model_path)
    # print(f'Best model checkpoint: {best_ckpt.name}')
    # model = type(model).load_from_checkpoint(best_ckpt)


if __name__ == "__main__":
    main()
