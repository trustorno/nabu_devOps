output "account_id" {
  value = "${data.aws_caller_identity.current.account_id}"
}

output "vpc_cidr" {
  value = "${aws_vpc.main.cidr_block}"
}

output "subnet1" {
  value = "${aws_subnet.public.cidr_block}"
}

output "subnet2" {
  value = "${aws_subnet.private.cidr_block}"
}

output "db_address" {
  value = "${aws_db_instance.db.address}"
}

output "redis_address" {
  value = "${aws_elasticache_cluster.redis.cache_nodes.0.address}"
}

output "web_url" {
  value = "http://${aws_alb.lb.dns_name}/"
}

output "aws_bucket_name" {
  value = "${aws_s3_bucket.s3.id}"
}
