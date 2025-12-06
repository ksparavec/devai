# Common outputs for networking
# Each cloud implementation provides these outputs

output "vpc_id" {
  description = "ID of the VPC/VNet"
  value       = ""
}

output "subnet_ids" {
  description = "IDs of the subnets"
  value       = []
}

output "security_group_id" {
  description = "ID of the security group"
  value       = ""
}
