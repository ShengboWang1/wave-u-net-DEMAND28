import argparse
import json
import os
import soundfile as sf
import librosa
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from util.utils import compute_STOI, compute_PESQ, eval_composite
from util.utils import initialize_config, load_checkpoint

"""
Parameters
"""
parser = argparse.ArgumentParser("Wave-U-Net: Speech Enhancement")
parser.add_argument("-C", "--config", type=str, required=True, help="Model and dataset for enhancement (*.json).")
parser.add_argument("-D", "--device", default="-1", type=str, help="GPU for speech enhancement. default: CPU")
parser.add_argument("-O", "--output_dir", type=str, required=True, help="Where are audio save.")
parser.add_argument("-M", "--model_checkpoint_path", type=str, required=True, help="Checkpoint.")
args = parser.parse_args()

"""
Preparation
"""
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
config = json.load(open(args.config))
model_checkpoint_path = args.model_checkpoint_path
output_dir = args.output_dir
assert os.path.exists(output_dir), "Enhanced directory should be exist."

"""
DataLoader
"""
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
dataloader = DataLoader(dataset=initialize_config(config["dataset"]), batch_size=1, num_workers=0)

"""
Model
"""
model = initialize_config(config["model"])
model.load_state_dict(load_checkpoint(model_checkpoint_path, device))
model.to(device)
model.eval()

"""
Enhancement
"""
sample_length = config["custom"]["sample_length"]


def evaluate(model, dataloader):
  total_stoi = 0.0
  total_ssnr = 0.0
  total_pesq = 0.0
  total_csig = 0.0
  total_cbak = 0.0
  total_covl = 0.0
  count = 0
  for mixture, clean, name in tqdm(dataloader):
    assert len(name) == 1, "Only support batch size is 1 in enhancement stage."
    name = name[0]
    padded_length = 0

    mixture = mixture.to(device)  # [1, 1, T]

    # The input of the model should be fixed length.
    if mixture.size(-1) % sample_length != 0:
        padded_length = sample_length - (mixture.size(-1) % sample_length)
        mixture1 = torch.cat([mixture, torch.zeros(1, 1, padded_length, device=device)], dim=-1)

    assert mixture1.size(-1) % sample_length == 0 and mixture1.dim() == 3
    mixture_chunks = list(torch.split(mixture1, sample_length, dim=-1))

    enhanced_chunks = []
    for chunk in mixture_chunks:
        enhanced_chunks.append(model(chunk).detach().cpu())

    enhanced = torch.cat(enhanced_chunks, dim=-1)  # [1, 1, T]
    enhanced = enhanced if padded_length == 0 else enhanced[:, :, :-padded_length]

    enhanced = enhanced.reshape(-1).numpy()

    output_path = os.path.join(output_dir, f"{name}.wav")
    sf.write(output_path, enhanced, 16000, 'PCM_24')
    #librosa.output.write_wav(output_path, enhanced, sr=16000)
    clean = clean.numpy().reshape(-1)
    mixture = mixture.cpu().numpy().reshape(-1)


    assert len(mixture) == len(enhanced) == len(clean)
    eval_metric = eval_composite(clean, enhanced, sr=16000)

    
    total_pesq += eval_metric['pesq']
    total_ssnr += eval_metric['ssnr']
    total_stoi += eval_metric['stoi']
    total_cbak += eval_metric['cbak']
    total_csig += eval_metric['csig']
    total_covl += eval_metric['covl']

    count += 1
    #print(count)

  return total_stoi / count, total_pesq / count, total_ssnr / count, total_csig / count, total_cbak / count, total_covl / count

avg_stoi, avg_pesq, avg_ssnr, avg_csig, avg_cbak, avg_covl = evaluate(model, dataloader)

#avg_stoi, avg_pesq, avg_ssnr, avg_cbak, avg_csig, avg_covl = eva_noisy(test_file_list_path)

#print('Avg_loss: {:.4f}'.format(avg_eval))
print('STOI: {:.4f}'.format(avg_stoi))
print('SSNR: {:.4f}'.format(avg_ssnr))
print('PESQ: {:.4f}'.format(avg_pesq))
print('CSIG: {:.4f}'.format(avg_csig))
print('CBAK: {:.4f}'.format(avg_cbak))
print('COVL: {:.4f}'.format(avg_covl))



            
                

