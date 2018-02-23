#TODO: Drop all hardcoded values to variables

provider "aws" {
  region = "${var.region}"
}


resource "aws_security_group" "allow_internal" {
  name = "tf-sg-${var.environment_name}"
  description = "Allow all internal inbound traffic"
  vpc_id = "${aws_vpc.main.id}"

  ingress {
    from_port = 0
    to_port = 0
    protocol = "-1"
    cidr_blocks = [
      "${aws_vpc.main.cidr_block}"
    ]
    security_groups = []
  }
  egress {
    from_port = 0
    to_port = 0
    protocol = "-1"
    cidr_blocks = [
      "0.0.0.0/0"
    ]
  }
}

resource "aws_security_group" "lb" {
  name = "tf-sg-lb-${var.environment_name}"
  description = "Security group for load balancer"
  vpc_id = "${aws_vpc.main.id}"

  ingress {
    from_port = 0
    to_port = 0
    protocol = "-1"
    cidr_blocks = [
      "${aws_vpc.main.cidr_block}",
      # TODO: CLIENT CIDR
      "0.0.0.0/0"
    ]
  }
  egress {
    from_port = 0
    to_port = 0
    protocol = "-1"
    cidr_blocks = [
      "0.0.0.0/0"
    ]
  }
}
//
//module "app" {
//  source = "app"
//  broker_url = "${aws_elasticache_cluster.redis.cache_nodes.0.address}"
//  db_host = "${aws.db.address}"
//  image = "${data.aws_ecr_repository.ic.repository_url}:${var.image_tag}"
//  s3_key = "${aws_iam_access_key.s3-user-key.id}"
//  s3_secret = "${aws_iam_access_key.s3-user-key.secret}"
//  s3_bucket_name = "${aws_s3_bucket.s3.id}"
//  environment_name = "${var.environment_name}"
//  ecs_iam_role = "${data.aws_iam_role.for_ecs.arn}"
//}
