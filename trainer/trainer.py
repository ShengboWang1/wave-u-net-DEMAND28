import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch

from trainer.base_trainer import BaseTrainer
from util.utils import compute_STOI, compute_PESQ, eval_composite
plt.switch_backend('agg')


class Trainer(BaseTrainer):
    def __init__(
            self,
            config,
            resume: bool,
            model,
            loss_function,
            optimizer,
            train_dataloader,
            validation_dataloader,
    ):
        super(Trainer, self).__init__(config, resume, model, loss_function, optimizer)
        self.train_data_loader = train_dataloader
        self.validation_data_loader = validation_dataloader

    def _train_epoch(self, epoch):
        loss_total = 0.0

        for i, (mixture, clean, name) in enumerate(self.train_data_loader):
            mixture = mixture.to(self.device)
            clean = clean.to(self.device)

            self.optimizer.zero_grad()
            enhanced = self.model(mixture)
            loss = self.loss_function(clean, enhanced)
            loss.backward()
            self.optimizer.step()

            loss_total += loss.item()

        dl_len = len(self.train_data_loader)
        self.writer.add_scalar(f"Train/Loss", loss_total / dl_len, epoch)

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        visualize_audio_limit = self.validation_custom_config["visualize_audio_limit"]
        visualize_waveform_limit = self.validation_custom_config["visualize_waveform_limit"]
        visualize_spectrogram_limit = self.validation_custom_config["visualize_spectrogram_limit"]

        sample_length = self.validation_custom_config["sample_length"]

        stoi_c_n = []  # clean and noisy
        stoi_c_e = []  # clean and enhanced
        pesq_c_n = []
        pesq_c_e = []
        # csig_c_n = []
        # csig_c_e = []
        # cbak_c_n = []
        # cbak_c_e = []
        # covl_c_n = []
        # covl_c_e = []
        # ssnr_c_n = []
        # ssnr_c_e = []

        for i, (mixture, clean, name) in enumerate(self.validation_data_loader):
            assert len(name) == 1, "Only support batch size is 1 in enhancement stage."
            name = name[0]
            padded_length = 0

            mixture = mixture.to(self.device)  # [1, 1, T]
            ######## 
            #print("mixture.size(-1) in validation")
            #print(mixture.size(-1))
            # The input of the model should be fixed length.
            ##################
            # add mixture1 instead of mixture to make sure the length are the same
            if mixture.size(-1) % sample_length != 0:
                padded_length = sample_length - (mixture.size(-1) % sample_length)
                mixture1 = torch.cat([mixture, torch.zeros(1, 1, padded_length, device=self.device)], dim=-1)

            assert mixture1.size(-1) % sample_length == 0 and mixture1.dim() == 3
            mixture_chunks = list(torch.split(mixture1, sample_length, dim=-1))
            enhanced_chunks = []
            for chunk in mixture_chunks:
                enhanced_chunks.append(self.model(chunk).detach().cpu())

            enhanced = torch.cat(enhanced_chunks, dim=-1)  # [1, 1, T]
            enhanced = enhanced if padded_length == 0 else enhanced[:, :, :-padded_length]

            enhanced = enhanced.reshape(-1).numpy()
            clean = clean.numpy().reshape(-1)
            mixture = mixture.cpu().numpy().reshape(-1)

            ##########
            # print("length of mixture enhanced clean in trainer.py")
            # print(len(mixture))
            # print(len(enhanced))
            # print(len(clean))
            #########
            assert len(mixture) == len(enhanced) == len(clean)

            # Visualize audio
            if i <= visualize_audio_limit:
                self.writer.add_audio(f"Speech/{name}_Noisy", mixture, epoch, sample_rate=16000)
                self.writer.add_audio(f"Speech/{name}_Enhanced", enhanced, epoch, sample_rate=16000)
                self.writer.add_audio(f"Speech/{name}_Clean", clean, epoch, sample_rate=16000)

            # Visualize waveform
            if i <= visualize_waveform_limit:
                fig, ax = plt.subplots(3, 1)
                for j, y in enumerate([mixture, enhanced, clean]):
                    ax[j].set_title("mean: {:.3f}, std: {:.3f}, max: {:.3f}, min: {:.3f}".format(
                        np.mean(y),
                        np.std(y),
                        np.max(y),
                        np.min(y)
                    ))
                    librosa.display.waveplot(y, sr=16000, ax=ax[j])
                plt.tight_layout()
                self.writer.add_figure(f"Waveform/{name}", fig, epoch)

            # Visualize spectrogram
            noisy_mag, _ = librosa.magphase(librosa.stft(mixture, n_fft=320, hop_length=160, win_length=320))
            enhanced_mag, _ = librosa.magphase(librosa.stft(enhanced, n_fft=320, hop_length=160, win_length=320))
            clean_mag, _ = librosa.magphase(librosa.stft(clean, n_fft=320, hop_length=160, win_length=320))

            if i <= visualize_spectrogram_limit:
                fig, axes = plt.subplots(3, 1, figsize=(6, 6))
                for k, mag in enumerate([
                    noisy_mag,
                    enhanced_mag,
                    clean_mag,
                ]):
                    axes[k].set_title(f"mean: {np.mean(mag):.3f}, "
                                      f"std: {np.std(mag):.3f}, "
                                      f"max: {np.max(mag):.3f}, "
                                      f"min: {np.min(mag):.3f}")
                    librosa.display.specshow(librosa.amplitude_to_db(mag), cmap="magma", y_axis="linear", ax=axes[k], sr=16000)
                plt.tight_layout()
                self.writer.add_figure(f"Spectrogram/{name}", fig, epoch)

            # Metric
            # eval_metric = eval_composite(clean, mixture, sr=16000)
            # eval_metric_enhanced = eval_composite(clean, enhanced, sr=16000)

            stoi_c_n.append(compute_STOI(clean, mixture, sr=16000))
            stoi_c_e.append(compute_STOI(clean, enhanced, sr=16000))
            pesq_c_n.append(compute_PESQ(clean, mixture, sr=16000))
            pesq_c_e.append(compute_PESQ(clean, enhanced, sr=16000))
            # csig_c_n.append(eval_metric['csig'])
            # csig_c_e.append(eval_metric_enhanced['csig'])
            # cbak_c_n.append(eval_metric['cbak'])
            # cbak_c_e.append(eval_metric_enhanced['cbak'])
            # covl_c_n.append(eval_metric['covl'])
            # covl_c_e.append(eval_metric_enhanced['covl'])
            # ssnr_c_n.append(eval_metric['ssnr'])
            # ssnr_c_e.append(eval_metric_enhanced['ssnr'])

        get_metrics_ave = lambda metrics: np.sum(metrics) / len(metrics)
        self.writer.add_scalars(f"Metric/STOI", {
            "Clean and noisy": get_metrics_ave(stoi_c_n),
            "Clean and enhanced": get_metrics_ave(stoi_c_e)
        }, epoch)
        self.writer.add_scalars(f"Metric/PESQ", {
            "Clean and noisy": get_metrics_ave(pesq_c_n),
            "Clean and enhanced": get_metrics_ave(pesq_c_e)
        }, epoch)
        # self.writer.add_scalars(f"Metric/CSIG", {
        #     "Clean and noisy": get_metrics_ave(csig_c_n),
        #     "Clean and enhanced": get_metrics_ave(csig_c_e)
        # }, epoch)
        # self.writer.add_scalars(f"Metric/CBAK", {
        #     "Clean and noisy": get_metrics_ave(cbak_c_n),
        #     "Clean and enhanced": get_metrics_ave(cbak_c_e)
        # }, epoch)
        # self.writer.add_scalars(f"Metric/COVL", {
        #     "Clean and noisy": get_metrics_ave(covl_c_n),
        #     "Clean and enhanced": get_metrics_ave(covl_c_e)
        # }, epoch)
        # self.writer.add_scalars(f"Metric/SSNR", {
        #     "Clean and noisy": get_metrics_ave(ssnr_c_n),
        #     "Clean and enhanced": get_metrics_ave(ssnr_c_e)
        # }, epoch)

        score = (get_metrics_ave(stoi_c_e) + self._transform_pesq_range(get_metrics_ave(pesq_c_e))) / 2
        return score
