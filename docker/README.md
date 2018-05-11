### Running the docker:

```sh
cd docker
#build the image
docker build -t docker_image_name .

#run the python code inside the docker container
#below parameters are passed to the python script
#gpu processing need to be run with the special nvidia runtime
docker run --runtime=nvidia docker_image_name 
      -i inS3bucketPath
      -r procS3bucketPath 
      -o outS3bucketPath 
      -d db_host
      -u db_user 
      -p db_pass 
      -n db_name
      
#example run
docker run --runtime=nvidia dimage1 
      -d dbtest.host123.us-east-2.rds.amazonaws.com
      -u rafal 
      -p pass123 
      -n postgres
```


### Notes

- inS3bucketPath, procS3bucketPath, outS3bucketPath need to be passed as a paths inside the S3 bucket if default values need to be appended. inS3bucketPath contains the raw output of the Kinesis stream. Script will process all the files which exists in this directory and move this to the procS3bucketPath. Morover labeled video will be saved to the outS3bucketPath. Default values:
    - S3 bucket = 's3://tf-bucket-dev/'
    - inS3bucketPath = 'VIDEOS/RAW/'
    - procS3bucketPath = 'VIDEOS/RAW_PROCESSED/'
    - outS3bucketPath = 'VIDEOS/OUT/'

- To install nvidia-docker follow the steps in [here](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)).
- There is a need to make sure that the created docker container on EC2 machines have access both to Aurora DB and S3 bucket.
- For the tests the Aurora DB was created using the same VPC for EC2.
- Then I followed the instructions listed [here](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/USER_VPC.Scenarios.html#USER_VPC.Scenario1).

Please suggest if there is a better, more efficient way...
