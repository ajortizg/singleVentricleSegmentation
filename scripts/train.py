import hydra
import os.path as osp
from omegaconf import DictConfig
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger

from svs.modules.transforms_factory import TransformsFactory


@hydra.main(config_path="../conf", config_name="train", version_base=None)
def main(config: DictConfig):

    pl.seed_everything(42, workers=True)

    model = hydra.utils.instantiate(config["model"])
    data = hydra.utils.instantiate(config["data"])

    trs = TransformsFactory(config["transforms"])
    data.set_train_transforms(trs.get_train_transforms())
    data.set_val_transforms(trs.get_val_transforms())

    logger = TensorBoardLogger(
        osp.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "tb_logs"),
        name=model.__class__.__name__
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=1000,
        min_epochs=1000,
        num_sanity_val_steps=-1,
        devices=config["gpu"],
        precision=32,
        logger=logger,
        log_every_n_steps=1,
        enable_model_summary=True
    )

    trainer.fit(model, data)


if __name__ == "__main__":
    main()
