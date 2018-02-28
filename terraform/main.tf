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
      "${aws_vpc.main.cidr_block}",
      "0.0.0.0/0"
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

module "app" {
  source = "app"
  s3_key = "${aws_iam_access_key.s3-user-key.id}"
  s3_secret = "${aws_iam_access_key.s3-user-key.secret}"
  s3_bucket_name = "${aws_s3_bucket.s3.id}"
  environment_name = "${var.environment_name}"
  ecs_iam_role = "${data.aws_iam_role.for_ecs.arn}"
  redis_host = "${aws_elasticache_cluster.redis.cache_nodes.0.address}"
  //  db_name = "${var.db_name}"
  //  db_user = "${var.db_user}"
  //  db_password = "${var.db_password}"
  db_host = "${aws_db_instance.db.address}"
  //  Temp image
  image = "nginx"
  cluster_id = "${aws_ecs_cluster.ecs-cluster-web.id}"
  target_lb_group_arn = "${aws_alb_target_group.lb-tg.arn}"
}
