resource "aws_cloudwatch_log_group" "log-group" {
  name = "tf-log-group-${var.environment_name}"
}

resource "aws_cloudwatch_log_stream" "log-stream" {
  name = "tf-log-stream-${var.environment_name}"
  log_group_name = "${aws_cloudwatch_log_group.log-group.name}"
}