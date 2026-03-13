# ─────────────────────────────────────────────────────────────
# EC2 Spot GPU instance for Ollama LLM inference
# ─────────────────────────────────────────────────────────────

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

# ── Data sources ──────────────────────────────────────────────

# NVIDIA Deep Learning AMI (Ubuntu 22.04) — comes with CUDA + drivers
data "aws_ami" "deep_learning" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning Base AMI with Single CUDA (Ubuntu 22.04) *"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }

  filter {
    name   = "state"
    values = ["available"]
  }
}

data "aws_vpc" "selected" {
  id      = var.vpc_id != "" ? var.vpc_id : null
  default = var.vpc_id == ""
}

data "aws_subnets" "available" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.selected.id]
  }

  filter {
    name   = "map-public-ip-on-launch"
    values = ["true"]
  }
}

# ── Security Group ────────────────────────────────────────────

resource "aws_security_group" "llm_server" {
  name_prefix = "${var.project_name}-llm-"
  description = "Security group for LLM inference server"
  vpc_id      = data.aws_vpc.selected.id

  # SSH
  ingress {
    description = "SSH access"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_ssh_cidrs
  }

  # Ollama API
  ingress {
    description = "Ollama API"
    from_port   = 11434
    to_port     = 11434
    protocol    = "tcp"
    cidr_blocks = var.allowed_api_cidrs
  }

  # Application API
  ingress {
    description = "Application API"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = var.allowed_api_cidrs
  }

  # Outbound — allow all
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-llm-sg"
  }

  lifecycle {
    create_before_destroy = true
  }
}

# ── EC2 Spot Instance ─────────────────────────────────────────

resource "aws_instance" "llm_server" {
  ami           = data.aws_ami.deep_learning.id
  instance_type = var.instance_type
  key_name      = var.key_pair_name
  subnet_id     = var.subnet_id != "" ? var.subnet_id : data.aws_subnets.available.ids[0]

  vpc_security_group_ids = [aws_security_group.llm_server.id]

  # Spot instance configuration
  instance_market_options {
    market_type = "spot"
    spot_options {
      max_price                      = var.spot_max_price
      spot_instance_type             = "persistent"
      instance_interruption_behavior = "stop"
    }
  }

  # Root volume — needs space for models
  root_block_device {
    volume_size           = var.root_volume_size
    volume_type           = "gp3"
    encrypted             = true
    delete_on_termination = true

    tags = {
      Name = "${var.project_name}-llm-root"
    }
  }

  # User data — setup script runs on first boot
  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    ollama_models = join(" ", var.ollama_models)
  }))

  # Allow instance to be stopped/started (spot persistent)
  instance_initiated_shutdown_behavior = "stop"

  tags = {
    Name = "${var.project_name}-llm-${var.environment}"
  }
}

# ── Elastic IP (survives spot stop/start) ─────────────────────

resource "aws_eip" "llm_server" {
  instance = aws_instance.llm_server.id
  domain   = "vpc"

  tags = {
    Name = "${var.project_name}-llm-eip"
  }
}
