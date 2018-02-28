resource "aws_vpc" "main" {
  cidr_block = "10.1.0.0/16"

  tags {
    Name = "vpc-${var.environment_name}"
  }
}

resource "aws_subnet" "public" {
  vpc_id = "${aws_vpc.main.id}"
  cidr_block = "10.1.0.0/24"
  availability_zone = "eu-west-1a"


  tags {
    Name = "public-${var.environment_name}"
  }
}

resource "aws_subnet" "private" {
  vpc_id = "${aws_vpc.main.id}"
  cidr_block = "10.1.1.0/24"
  availability_zone = "eu-west-1b"

  tags {
    Name = "private-${var.environment_name}"
  }
}

resource "aws_internet_gateway" "igw" {
  vpc_id = "${aws_vpc.main.id}"
  tags {
    Name = "igw-${var.environment_name}"
  }
}

resource "aws_eip" "nat" {
  vpc = true
  tags {
    Name = "eip-nat-${var.environment_name}"
  }
}

resource "aws_nat_gateway" "nat" {
  allocation_id = "${aws_eip.nat.id}"
  subnet_id = "${aws_subnet.public.id}"

  depends_on = [
    "aws_internet_gateway.igw"
  ]
  tags {
    Name = "nat-${var.environment_name}"
  }

}


resource "aws_route_table" "rt-public" {
  vpc_id = "${aws_vpc.main.id}"

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = "${aws_internet_gateway.igw.id}"
  }
  tags {
    Name = "tf-public-${var.environment_name}"
  }
}

resource "aws_route_table_association" "rt-association-public" {
  subnet_id = "${aws_subnet.public.id}"
  route_table_id = "${aws_route_table.rt-public.id}"
}

resource "aws_route_table" "rt-nat" {
  vpc_id = "${aws_vpc.main.id}"

  route {
    cidr_block = "0.0.0.0/0"
    nat_gateway_id = "${aws_nat_gateway.nat.id}"
  }

  tags {
    Name = "tf-nat-${var.environment_name}"
  }
}

resource "aws_route_table_association" "rt-association-nat" {
  subnet_id = "${aws_subnet.private.id}"
  route_table_id = "${aws_route_table.rt-nat.id}"
}

resource "aws_alb_target_group" "lb-tg" {
  name = "tf-lb-tg-${var.environment_name}"
  port = 80
  protocol = "HTTP"
  vpc_id = "${aws_vpc.main.id}"
  depends_on = [
    "aws_alb.lb"
  ]
}

resource "aws_alb" "lb" {
  name = "tf-lb-${var.environment_name}"
  internal = false
  security_groups = [
    "${aws_security_group.lb.id}"
  ]

  subnets = [
    "${aws_subnet.public.id}",
    "${aws_subnet.private.id}",
  ]
}

resource "aws_alb_listener" "lb-listener" {
  load_balancer_arn = "${aws_alb.lb.id}"
  port = "80"
  protocol = "HTTP"

  default_action {
    target_group_arn = "${aws_alb_target_group.lb-tg.id}"
    type = "forward"
  }
}