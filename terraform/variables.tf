variable "environment_name" {
  default = "dev"
  description = "Postfix name for all components"
}

variable "region" {
  default = "eu-west-1"
}

variable "ssh_key_name" {
}

variable "vpc_id" {
}

variable "db_storage" {
  default = "30"
  description = "Storage size in GB"
}

variable "db_instance_type" {
  default = "db.t2.micro"
  description = "Instance class"
}

variable "db_name" {
  default = "nabu"
  description = "db name"
}

variable "db_user" {
  default = "nabu"
  description = "User name"
}

variable "db_password" {
  description = "DB password, provide through your ENV variables, should be more than 8 letters"
}

variable "ecs_ami_fixed" {
  default = "ami-d65dfbaf"
  description = "Hardcoded ami for ecs"
}
variable "ecs_ami_instance_type" {
  default = "t2.micro"
}

variable "redis_instance_type" {
  default = "cache.t2.micro"
}

variable "image_tag" {
  default = "latest"
}

variable "dynamo_read_capacity" {
  default = 5
}

variable "dynamo_write_capacity" {
  default = 5
}

variable "ecs_desired_capacity" {
  default = 3
}
variable "ecs_min_capacity" {
  default = 2
}
variable "ecs_max_capacity" {
  default = 5
}