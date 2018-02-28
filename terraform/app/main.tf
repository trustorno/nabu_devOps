data "template_file" "basic" {
  template = "${file("app/templates/config.json")}"

  vars {
    image = "${var.image}"
    s3_key = "${var.s3_key}"
    s3_secret = "${var.s3_secret}"
    s3_bucket_name = "${var.s3_bucket_name}"
    redis_host = "${var.redis_host}"
    db_host = "${var.db_host}"
  }
}

data "template_file" "web" {
  template = "${data.template_file.basic.rendered}"

  vars {
    name = "web"
    command = ""
  }
}
resource "aws_ecs_task_definition" "web" {
  family = "tf-web-${var.environment_name}"
  container_definitions = "${data.template_file.web.rendered}"
}
resource "aws_ecs_service" "web" {
  name = "tf-web-service-${var.environment_name}"
  cluster = "${var.cluster_id}"
  task_definition = "${aws_ecs_task_definition.web.arn}"
  desired_count = 1
  iam_role = "${var.ecs_iam_role}"

  load_balancer {
    target_group_arn = "${var.target_lb_group_arn}"
    container_name = "web"
    container_port = 80
  }
}


//data "template_file" "app" {
//  template = "${data.template_file.basic.rendered}"
//
//  vars {
//    name = "app"
//    command = "app"
//  }
//}
//resource "aws_ecs_task_definition" "app" {
//  family = "tf-app-${var.environment_name}"
//  container_definitions = "${data.template_file.app.rendered}"
//}
//resource "aws_ecs_service" "app" {
//  name = "tf-app-service-${var.environment_name}"
//  cluster = "${var.cluster_id}"
//  task_definition = "${aws_ecs_task_definition.app.arn}"
//  desired_count = 1
//  iam_role = "${var.ecs_iam_role}"
//
//}