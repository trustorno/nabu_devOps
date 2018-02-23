resource "aws_iam_role" "ecsInstanceRole" {
  name = "ecsInstanceRole-${var.environment_name}"

  assume_role_policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Action": "sts:AssumeRole",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Effect": "Allow",
      "Sid": ""
    }
  ]
}
EOF
}

resource "aws_iam_instance_profile" "ecsInstanceProfile" {
  name = "tf-ecsInstanceProfile-${var.environment_name}"
  role = "${aws_iam_role.ecsInstanceRole.name}"
}

resource "aws_iam_role_policy_attachment" "ecsInstanceRole" {
  role = "${aws_iam_role.ecsInstanceRole.name}"
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}


data "template_file" "ecs_init_launch" {
  template = "${file("templates/ecs_config.sh")}"

  vars {
    cluster_name = "${aws_ecs_cluster.ecs-cluster-web.name}"
  }
}

resource "aws_launch_configuration" "ecs-ec2-instance" {
  name = "tf-ec2-ecs-${var.environment_name}"
  security_groups = [
    "${aws_security_group.allow_internal.id}",
  ]

  key_name = "${var.ssh_key_name}"
  image_id = "${var.ecs_ami_fixed}"
  instance_type = "${var.ecs_ami_instance_type}"
  associate_public_ip_address = true
  user_data = "${data.template_file.ecs_init_launch.rendered}"
  # IAM role
  iam_instance_profile = "${aws_iam_instance_profile.ecsInstanceProfile.name}"

  # aws_launch_configuration can not be modified.
  # Therefore we use create_before_destroy so that a new modified aws_launch_configuration can be created
  # before the old one get's destroyed. That's why we use name_prefix instead of name.
  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_autoscaling_group" "ecs-pool-web" {
  name = "tf-ecs-pool-web-${var.environment_name}"
  vpc_zone_identifier = [
    "${aws_subnet.public.id}"
  ]
  max_size = "${var.ecs_max_capacity}"
  min_size = "${var.ecs_min_capacity}"
  desired_capacity = "${var.ecs_desired_capacity}"
  force_delete = true
  launch_configuration = "${aws_launch_configuration.ecs-ec2-instance.name}"

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_ecs_cluster" "ecs-cluster-web" {
  name = "tf-ecs-web-${var.environment_name}"
}


data "template_file" "ecs_init_launch-app" {
  template = "${file("templates/ecs_config.sh")}"

  vars {
    cluster_name = "${aws_ecs_cluster.ecs-cluster-app.name}"
  }
}

resource "aws_launch_configuration" "ecs-ec2-instance-app" {
  name = "tf-ec2-ecs-app-${var.environment_name}"
  security_groups = [
    "${aws_security_group.allow_internal.id}",
  ]

  key_name = "${var.ssh_key_name}"
  image_id = "${var.ecs_ami_fixed}"
  instance_type = "${var.ecs_ami_instance_type}"
  associate_public_ip_address = true
  user_data = "${data.template_file.ecs_init_launch.rendered}"
  # IAM role
  iam_instance_profile = "${aws_iam_instance_profile.ecsInstanceProfile.name}"

  # aws_launch_configuration can not be modified.
  # Therefore we use create_before_destroy so that a new modified aws_launch_configuration can be created
  # before the old one get's destroyed. That's why we use name_prefix instead of name.
  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_autoscaling_group" "ecs-pool-app" {
  name = "tf-ecs-pool-app-${var.environment_name}"
  vpc_zone_identifier = [
    "${aws_subnet.public.id}"
  ]
  max_size = "${var.ecs_max_capacity}"
  min_size = "${var.ecs_min_capacity}"
  desired_capacity = "${var.ecs_desired_capacity}"
  force_delete = true
  launch_configuration = "${aws_launch_configuration.ecs-ec2-instance-app.name}"

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_ecs_cluster" "ecs-cluster-app" {
  name = "tf-ecs-app-${var.environment_name}"
}