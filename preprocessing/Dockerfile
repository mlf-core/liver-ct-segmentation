FROM jupyter/minimal-notebook:latest

COPY environment.yml .

# Create the environment
RUN conda env create -f environment.yml && conda clean -a

# Activate the environment
RUN echo "source activate liver_ct_seg_data" >> ~/.bashrc
ENV PATH /home/user/miniconda/envs/liver_ct_seg_data/bin:$PATH

# Dump the details of the installed packages to a file for posterity
RUN conda env export --name liver_ct_seg_data > liver_ct_seg_data_environment.yml

COPY start.sh .
Copy liver_ct_seg_dataset.ipynb .
