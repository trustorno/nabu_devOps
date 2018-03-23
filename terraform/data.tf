data "aws_caller_identity" "current" {}

//data "aws_vpc" "vpc" {
//  id = "${var.vpc_id}"
//}
//
//data "aws_subnet" "subnet" {
//  filter {
//    name = "vpc-id"
//    values = [
//      "${var.vpc_id}"
//    ]
//  }
//
//  filter {
//    name = "tag:Name"
//    values = [
//      "default"
//    ]
//  }
//}
//
//data "aws_subnet" "subnet2" {
//  filter {
//    name = "vpc-id"
//    values = [
//      "${var.vpc_id}"
//    ]
//  }
//
//  filter {
//    name = "tag:Name"
//    values = [
//      "second"
//    ]
//  }
//}

data "aws_ami" "ecs_ami-auto" {
  most_recent = true

  filter {
    name = "architecture"
    values = [
      "x86_64"]
  }

  filter {
    name = "name"
    values = [
      "amzn-ami-*-amazon-ecs-optimized"]
  }

  filter {
    name = "owner-alias"
    values = [
      "amazon"]
  }

  filter {
    name = "root-device-type"
    values = [
      "ebs"]
  }

  filter {
    name = "virtualization-type"
    values = [
      "hvm"]
  }
}

variable "aws_ami_const" {
  default = "ami-0693ed7f"
}
data "aws_iam_role" "for_ecs" {
  name = "AWSServiceRoleForECS"
}

data "aws_ecr_repository" "image" {
  name = "nabu"
}


