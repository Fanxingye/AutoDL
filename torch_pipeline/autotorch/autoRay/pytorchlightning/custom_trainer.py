from typing import Optional, Union
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.trainer.states import TrainerFn, TrainerStatus
from pytorch_lightning.utilities.types import _EVALUATE_OUTPUT, _PREDICT_OUTPUT, EVAL_DATALOADERS, TRAIN_DATALOADERS
from pytorch_lightning.utilities import rank_zero_deprecation
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class CustomTrainer(Trainer):
    def fit(
            self,
            model: "pl.LightningModule",
            train_dataloaders: Optional[Union[TRAIN_DATALOADERS,
                                              LightningDataModule]] = None,
            val_dataloaders: Optional[EVAL_DATALOADERS] = None,
            datamodule: Optional[LightningDataModule] = None,
            train_dataloader=None,  # noqa TODO: remove with 1.6
    ) -> _EVALUATE_OUTPUT:
        r"""
        Runs the full optimization routine.

        Args:
            model: Model to fit.

            train_dataloaders: A collection of :class:`torch.utils.data.DataLoader` or a
                :class:`~pytorch_lightning.core.datamodule.LightningDataModule` specifying training samples.
                In the case of multiple dataloaders, please see this :ref:`page <multiple-training-dataloaders>`.

            val_dataloaders: A :class:`torch.utils.data.DataLoader` or a sequence of them specifying validation samples.

            datamodule: An instance of :class:`~pytorch_lightning.core.datamodule.LightningDataModule`.
        """
        Trainer._log_api_event("fit")

        self.state.fn = TrainerFn.FITTING
        self.state.status = TrainerStatus.RUNNING
        self.training = True

        if train_dataloader is not None:
            rank_zero_deprecation(
                "`trainer.fit(train_dataloader)` is deprecated in v1.4 and will be removed in v1.6."
                " Use `trainer.fit(train_dataloaders)` instead. HINT: added 's'"
            )
            train_dataloaders = train_dataloader
        # if a datamodule comes in as the second arg, then fix it for the user
        if isinstance(train_dataloaders, LightningDataModule):
            datamodule = train_dataloaders
            train_dataloaders = None
        # If you supply a datamodule you can't supply train_dataloader or val_dataloaders
        if (train_dataloaders is not None
                or val_dataloaders is not None) and datamodule is not None:
            raise MisconfigurationException(
                "You cannot pass `train_dataloader` or `val_dataloaders` to `trainer.fit(datamodule=...)`"
            )

        # links data to the trainer
        self.data_connector.attach_data(model,
                                        train_dataloaders=train_dataloaders,
                                        val_dataloaders=val_dataloaders,
                                        datamodule=datamodule)

        self.checkpoint_connector.resume_start()

        results = self._run(model)

        assert self.state.stopped
        self.training = False

        return results
