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
      -o outS3bucketPath 
      -d db_host
      -u db_user 
      -p db_pass 
      -n db_name
      
#example run
docker run --runtime=nvidia dimage1 
      -i POC/VIDEO/test4.mp4
      -o OUTPUT/out1.avi 
      -d dbtest.host123.us-east-2.rds.amazonaws.com
      -u rafal 
      -p pass123 
      -n postgres
```


### Notes

- IN_S3_path, OUT_S3_path need to be passed as a paths inside the S3 bucket. Default S3 bucket is set to 's3://tf-bucket-dev/'. This can be changed with the -t parameter
- To install nvidia-docker follow the steps in [here](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)).
- There is a need to make sure that the created docker container on EC2 machines have access both to Aurora DB and S3 bucket.
- For the tests the Aurora DB was created using the same VPC for EC2.
- Then I followed the instructions listed [here](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/USER_VPC.Scenarios.html#USER_VPC.Scenario1).

Please suggest if there is a better, more efficient way...
