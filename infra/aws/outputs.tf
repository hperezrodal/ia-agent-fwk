# ─────────────────────────────────────────────────────────────
# Outputs — displayed after terraform apply
# ─────────────────────────────────────────────────────────────

output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.llm_server.id
}

output "public_ip" {
  description = "Elastic IP address"
  value       = aws_eip.llm_server.public_ip
}

output "ssh_command" {
  description = "SSH command to connect"
  value       = "ssh -i ~/.ssh/${var.key_pair_name}.pem ubuntu@${aws_eip.llm_server.public_ip}"
}

output "ollama_url" {
  description = "Ollama API endpoint"
  value       = "http://${aws_eip.llm_server.public_ip}:11434"
}

output "ami_id" {
  description = "AMI ID used (Deep Learning AMI)"
  value       = data.aws_ami.deep_learning.id
}

output "ami_name" {
  description = "AMI name"
  value       = data.aws_ami.deep_learning.name
}

output "spot_price" {
  description = "Max spot price configured"
  value       = var.spot_max_price
}
