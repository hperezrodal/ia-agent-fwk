# ─────────────────────────────────────────────────────────────
# Variables — EC2 Spot GPU instance for LLM inference
# ─────────────────────────────────────────────────────────────

variable "aws_region" {
  description = "AWS region to deploy in"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name used for resource naming and tagging"
  type        = string
  default     = "ia-agent-fwk"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "staging"
}

# ── Instance ──────────────────────────────────────────────────

variable "instance_type" {
  description = "EC2 instance type. g5.xlarge = 1x A10G 24GB, g4dn.xlarge = 1x T4 16GB"
  type        = string
  default     = "g5.xlarge"
}

variable "spot_max_price" {
  description = "Maximum hourly price for spot instance (USD). Empty = on-demand price cap."
  type        = string
  default     = "0.50"
}

variable "root_volume_size" {
  description = "Root EBS volume size in GB (models + Docker images need space)"
  type        = number
  default     = 100
}

variable "key_pair_name" {
  description = "Name of the SSH key pair to use. Must already exist in the region."
  type        = string
}

# ── Network ───────────────────────────────────────────────────

variable "allowed_ssh_cidrs" {
  description = "CIDR blocks allowed for SSH access"
  type        = list(string)
  default     = [] # Must be set — no default for security
}

variable "allowed_api_cidrs" {
  description = "CIDR blocks allowed for Ollama API (port 11434) and app API (port 8000)"
  type        = list(string)
  default     = [] # Must be set
}

variable "vpc_id" {
  description = "VPC ID to deploy into. Empty = use default VPC."
  type        = string
  default     = ""
}

variable "subnet_id" {
  description = "Subnet ID to deploy into. Empty = use first available subnet."
  type        = string
  default     = ""
}

# ── LLM ───────────────────────────────────────────────────────

variable "ollama_models" {
  description = "List of Ollama models to pull on first boot"
  type        = list(string)
  default     = ["qwen3:30b-a3b-q4_K_M", "nomic-embed-text"]
}
