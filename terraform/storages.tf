resource "aws_elasticache_subnet_group" "redis" {
  name = "tf-redis-subnet-${var.environment_name}"
  subnet_ids = [
    "${aws_subnet.public.id}"
  ]
}

resource "aws_elasticache_cluster" "redis" {
  cluster_id = "tf-redis-${var.environment_name}"

  engine = "redis"
  engine_version = "3.2.10"
  node_type = "${var.redis_instance_type}"
  port = 6379
  num_cache_nodes = 1
  parameter_group_name = "default.redis3.2"
  security_group_ids = [
    "${aws_security_group.allow_internal.id}"
  ]
  subnet_group_name = "${aws_elasticache_subnet_group.redis.name}"
}


resource "aws_db_subnet_group" "db" {
  name = "tf-rds-subnet-${var.environment_name}"
  subnet_ids = [
    "${aws_subnet.public.id}",
    "${aws_subnet.private.id}",
  ]
}

resource "aws_db_instance" "db" {
  identifier = "tf-rds-${var.environment_name}"
  allocated_storage = "${var.db_storage}"
  engine = "postgres"
  engine_version = "9.6.6"
  instance_class = "${var.db_instance_type}"
  name = "${var.db_name}"
  username = "${var.db_user}"
  password = "${var.db_password}"
  vpc_security_group_ids = [
    "${aws_security_group.allow_internal.id}"
  ]
  db_subnet_group_name = "${aws_db_subnet_group.db.id}"
  multi_az = false
  # TODO: do something with snapshot
  skip_final_snapshot = true
}


resource "aws_s3_bucket" "s3" {
  bucket = "tf-bucket-${var.environment_name}"
  acl = "private"
}

resource "aws_iam_user" "s3-user" {
  name = "tf-api-user-s3-${var.environment_name}"
}

resource "aws_iam_access_key" "s3-user-key" {
  user = "${aws_iam_user.s3-user.name}"
}

resource "aws_iam_user_policy_attachment" "s3-user-policy" {
  user = "${aws_iam_user.s3-user.name}"
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}


