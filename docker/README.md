### Running the docker:

```sh
cd docker
#build the image
docker build -t docker_image_name .

#run the python code inside the docker container
#below parameters are passed to the python script
docker run docker_image_name -i s3://tf-bucket-dev/VIDEOS/test1.mp4 
                              -o s3://tf-bucket-dev/OUTPUT/output_video.mp4 
                              -d aurora_db_host_name 
                              -u username 
                              -p password 
                              -n db_name

```


### Notes

- There is a need to make sure that the created docker container on EC2 machines have access both to Aurora DB and S3 bucket.
- For the tests the Aurora DB was created using the same VPC for EC2.
- Then I followed the instructions listed [here](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/USER_VPC.Scenarios.html#USER_VPC.Scenario1).

Please suggest if there is a better, more efficient way...
