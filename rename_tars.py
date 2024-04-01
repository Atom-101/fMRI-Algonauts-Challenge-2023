import tarfile
import os
import shutil
from tqdm import tqdm

def rename_files_in_tar(tar_filename, file_mapping):
    # Create a temporary tar file for the renamed archive
    temp_tar_filename = '/tmp/' + tar_filename.split('/')[-1]
    
    # Open the original tar archive for reading
    with tarfile.open(tar_filename, 'r') as original_tar:
        # Create a new tar archive for writing with the temporary file
        pngs = {}
        with tarfile.open(temp_tar_filename, 'w') as temp_tar:
            for member in tqdm(sorted(original_tar.getmembers(), key=lambda x:x.name)):
                # Check if the file needs to be renamed
                new_name = member.name  # file_mapping(member.name)
                if 'png' in new_name:
                    pngs[new_name] = member
            
            for member in tqdm(sorted(original_tar.getmembers(), key=lambda x:x.name)):
                if 'eeg.npy' in member.name:
                    temp_tar.addfile(member, original_tar.extractfile(member))
                    png_name = member.name[:-7] + 'png'
                    temp_tar.addfile(pngs[png_name], original_tar.extractfile(pngs[png_name]))
    
    # Replace the original tar file with the temporary one
    os.remove(tar_filename)
    shutil.move(temp_tar_filename, tar_filename)

def rename_func(s):
    s = s.split('.')
    s[0] = '_'.join(s[:2])
    s.pop(1)
    return '.'.join(s)

if __name__ == "__main__":
    tar_filename = "/fsx/proj-fmri/shared/eeg_mnist_train.tar"  # Replace with your tar archive filename
    # tar_filename = "/fsx/proj-fmri/shared/eeg_mnist_test.tar"  # Replace with your tar archive filename
    file_mapping = rename_func

    rename_files_in_tar(tar_filename, file_mapping)
