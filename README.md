# sagemaker-spam-classification
Spam classification pipeline at Amazon SageMaker. Components of the pipeline are data processing, model training and model serving.

## Setup and running in Local Mode

Requirements to run a pipeline locally:
* python 3.7
* Docker
* AWS CLI

Configure AWS credentials and region:
```bash
aws configure
```

Install all required packages:
```bash
pip install -r requirements.txt
```

To run a pipeline in Local Mode use `local_pipeline.ipynb` notebook.

## Setup and running at Amazon SageMaker

To set up a pipeline at Amazon SageMaker:
1. Create a notebook instance and associate a git repository with the instance.
2. Use `pipeline.ipynb` notebook to run components.
 