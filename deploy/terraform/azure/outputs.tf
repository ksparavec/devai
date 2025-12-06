# Azure Terraform outputs

output "container_fqdn" {
  description = "FQDN of the container instance"
  value       = azurerm_container_group.main.fqdn
}

output "service_url" {
  description = "URL to access JupyterLab"
  value       = "http://${azurerm_container_group.main.fqdn}:8888"
}

output "container_ip" {
  description = "Public IP of the container instance"
  value       = azurerm_container_group.main.ip_address
}

output "acr_login_server" {
  description = "ACR login server"
  value       = azurerm_container_registry.main.login_server
}

output "acr_name" {
  description = "ACR name"
  value       = azurerm_container_registry.main.name
}

output "resource_group_name" {
  description = "Resource group name"
  value       = azurerm_resource_group.main.name
}

output "push_command" {
  description = "Command to push image to ACR"
  sensitive   = true
  value       = <<-EOT
    # Login to ACR
    az acr login --name ${azurerm_container_registry.main.name}

    # Build and push
    docker build -t ${var.project_name} .
    docker tag ${var.project_name}:latest ${azurerm_container_registry.main.login_server}/${var.project_name}:latest
    docker push ${azurerm_container_registry.main.login_server}/${var.project_name}:latest
  EOT
}
