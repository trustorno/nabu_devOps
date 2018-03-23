resource "aws_kinesis_stream" "test_stream" {
  name = "tf-${var.environment_name}"
  shard_count = 1
  retention_period = 12

  shard_level_metrics = [
    "IncomingBytes",
    "OutgoingBytes",
  ]

}