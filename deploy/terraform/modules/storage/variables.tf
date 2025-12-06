# Common interface for storage across all clouds

variable "project_name" {
  description = "Name of the project"
  type        = string
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
}

variable "storage_size_gb" {
  description = "Storage size in GB"
  type        = number
  default     = 50
}

variable "subnet_ids" {
  description = "Subnet IDs for the storage resources"
  type        = list(string)
  default     = []
}

variable "security_group_id" {
  description = "Security group ID"
  type        = string
  default     = ""
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}
