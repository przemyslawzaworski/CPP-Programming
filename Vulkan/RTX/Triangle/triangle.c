// cl.exe /EHsc /I"%VK_SDK_PATH%\include" triangle.c /link /LIBPATH:"%VK_SDK_PATH%\lib" user32.lib kernel32.lib gdi32.lib vulkan-1.lib

#include <windows.h>
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

PFN_vkCmdTraceRaysNV vkCmdTraceRays;
PFN_vkBindAccelerationStructureMemoryNV vkBindAccelerationStructureMemory;
PFN_vkCmdBuildAccelerationStructureNV vkCmdBuildAccelerationStructure;
PFN_vkCmdWriteAccelerationStructuresPropertiesNV vkCmdWriteAccelerationStructuresProperties;
PFN_vkCreateAccelerationStructureNV vkCreateAccelerationStructure;
PFN_vkCreateRayTracingPipelinesNV vkCreateRayTracingPipelines;
PFN_vkDestroyAccelerationStructureNV vkDestroyAccelerationStructure;
PFN_vkGetAccelerationStructureHandleNV vkGetAccelerationStructureHandle;
PFN_vkGetAccelerationStructureMemoryRequirementsNV vkGetAccelerationStructureMemoryRequirements;
PFN_vkGetRayTracingShaderGroupHandlesNV vkGetRayTracingShaderGroupHandles;

void vkInit(VkInstance instance)
{
	vkCmdTraceRays = (PFN_vkCmdTraceRaysNV)(vkGetInstanceProcAddr(instance, "vkCmdTraceRaysNV"));
	vkBindAccelerationStructureMemory = (PFN_vkBindAccelerationStructureMemoryNV)(vkGetInstanceProcAddr(instance, "vkBindAccelerationStructureMemoryNV"));
	vkCmdBuildAccelerationStructure = (PFN_vkCmdBuildAccelerationStructureNV)(vkGetInstanceProcAddr(instance, "vkCmdBuildAccelerationStructureNV"));
	vkCmdWriteAccelerationStructuresProperties = (PFN_vkCmdWriteAccelerationStructuresPropertiesNV)(vkGetInstanceProcAddr(instance, "vkCmdWriteAccelerationStructuresPropertiesNV"));
	vkCreateAccelerationStructure = (PFN_vkCreateAccelerationStructureNV)(vkGetInstanceProcAddr(instance, "vkCreateAccelerationStructureNV"));
	vkCreateRayTracingPipelines = (PFN_vkCreateRayTracingPipelinesNV)(vkGetInstanceProcAddr(instance, "vkCreateRayTracingPipelinesNV"));
	vkDestroyAccelerationStructure = (PFN_vkDestroyAccelerationStructureNV)(vkGetInstanceProcAddr(instance, "vkDestroyAccelerationStructureNV"));
	vkGetAccelerationStructureHandle = (PFN_vkGetAccelerationStructureHandleNV)(vkGetInstanceProcAddr(instance, "vkGetAccelerationStructureHandleNV"));
	vkGetAccelerationStructureMemoryRequirements = (PFN_vkGetAccelerationStructureMemoryRequirementsNV)(vkGetInstanceProcAddr(instance, "vkGetAccelerationStructureMemoryRequirementsNV"));
	vkGetRayTracingShaderGroupHandles = (PFN_vkGetRayTracingShaderGroupHandlesNV)(vkGetInstanceProcAddr(instance, "vkGetRayTracingShaderGroupHandlesNV"));
}

const uint32_t ScreenWidth = 1280;
const uint32_t ScreenHeight = 720;

struct BufferVulkan
{
	VkDeviceSize size;
	VkBuffer buffer;
	VkDeviceMemory memory;
};

struct ImageVulkan
{
	VkImageType type;
	VkFormat format;
	VkExtent3D extent;
	VkImage image;
	VkDeviceMemory memory;
	VkImageView view;
};

struct AccelerationStructure
{
	VkDeviceMemory memory;
	VkAccelerationStructureInfoNV accelerationStructureInfo;
	VkAccelerationStructureNV accelerationStructure;
	uint64_t handle;
};

struct VkGeometryInstance
{
	float transform[12];
	uint32_t instanceId : 24;
	uint32_t mask : 8;
	uint32_t instanceOffset : 24;
	uint32_t flags : 8;
	uint64_t accelerationStructureHandle;
};

VkDevice device;
VkQueue queue;
VkSwapchainKHR swapchain;
VkInstance instance;
VkPhysicalDevice physicalDevice;
VkSurfaceKHR surface;
VkCommandPool commandPool;
VkImage* swapchainImages;
VkImageView* swapchainImageViews;
VkFence* waitForFrameFences;
VkCommandBuffer* commandBuffers;
VkSemaphore semaphoreImageAcquired;
VkSemaphore semaphoreRenderFinished;
VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties;
VkDescriptorSet descriptorSet;
VkDescriptorPool descriptorPool;
VkDescriptorSetLayout descriptorSetLayout;
VkPipeline rtPipeline;
VkPipelineLayout pipelineLayout;
struct AccelerationStructure blas;
struct AccelerationStructure tlas;
uint32_t imageCount;

uint32_t GetMemoryType(VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties, VkMemoryRequirements memoryRequirements, VkMemoryPropertyFlags memoryProperties)
{
	uint32_t result = 0;
	for (uint32_t i = 0; i < VK_MAX_MEMORY_TYPES; i++)
	{
		if (memoryRequirements.memoryTypeBits & (1 << i))
		{
			if ((physicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & memoryProperties) == memoryProperties)
			{
				result = i;
				break;
			}
		}
	}
	return result;
}

void LoadBuffer(VkDevice device, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memoryProperties, void* pSrc, struct BufferVulkan* pBuffer)
{
	pBuffer->size = size;
	VkBufferCreateInfo pCreateInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, 0, 0, pBuffer->size, usage, VK_SHARING_MODE_EXCLUSIVE, 0, 0};
	vkCreateBuffer(device, &pCreateInfo, 0, &pBuffer->buffer);
	VkMemoryRequirements pMemoryRequirements;
	vkGetBufferMemoryRequirements(device, pBuffer->buffer, &pMemoryRequirements);
	uint32_t memoryTypeIndex = GetMemoryType(physicalDeviceMemoryProperties, pMemoryRequirements, memoryProperties);
	VkMemoryAllocateInfo pAllocateInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, 0, pMemoryRequirements.size, memoryTypeIndex };
	vkAllocateMemory(device, &pAllocateInfo, 0, &pBuffer->memory);
	vkBindBufferMemory(device, pBuffer->buffer, pBuffer->memory, 0);
	if (pSrc)
	{
		void* ppData;
		vkMapMemory(device, pBuffer->memory, 0, pBuffer->size, 0, &ppData);
		memcpy(ppData, pSrc, pBuffer->size);
		vkUnmapMemory(device, pBuffer->memory);
	}
}

void ReleaseBuffer(VkDevice device, struct BufferVulkan pBuffer)
{
	vkDestroyBuffer(device, pBuffer.buffer, 0);
	vkFreeMemory(device, pBuffer.memory, 0);
	ZeroMemory(&pBuffer, sizeof(struct BufferVulkan)); 
}

void LoadShader(VkDevice device, const char* filename, VkShaderModule* pShaderModule)
{
	FILE *file = fopen(filename, "rb");
	fseek(file, 0, SEEK_END);
	size_t size = ftell(file);
	rewind(file);
	char *source = (char*) malloc(size);
	fread(source, sizeof(char), size, file);
	fclose(file);
	VkShaderModuleCreateInfo pCreateInfo = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, 0, 0, size, (uint32_t*)(source)};
	vkCreateShaderModule(device, &pCreateInfo, 0, pShaderModule);
	free(source);
}

void LoadAccelerationStructure(VkDevice device, VkAccelerationStructureTypeNV type, uint32_t geometryCount, uint32_t instanceCount, struct VkGeometryNV* pGeometries, struct AccelerationStructure* pAccelerationStructure)
{
	VkAccelerationStructureInfoNV info = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV };
	info.type = type;
	info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_NV;
	info.geometryCount = geometryCount;
	info.instanceCount = instanceCount;
	info.pGeometries = pGeometries;
	VkAccelerationStructureCreateInfoNV pCreateInfo = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_NV, 0, 0, info };
	vkCreateAccelerationStructure(device, &pCreateInfo, 0, &pAccelerationStructure->accelerationStructure);
	VkAccelerationStructureMemoryRequirementsInfoNV memoryRequirementsInfo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_NV};
	memoryRequirementsInfo.type = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_OBJECT_NV;
	memoryRequirementsInfo.accelerationStructure = pAccelerationStructure->accelerationStructure;
	VkMemoryRequirements2 pMemoryRequirements;
	vkGetAccelerationStructureMemoryRequirements(device, &memoryRequirementsInfo, &pMemoryRequirements);
	VkMemoryAllocateInfo memoryAllocateInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
	memoryAllocateInfo.allocationSize = pMemoryRequirements.memoryRequirements.size;
	memoryAllocateInfo.memoryTypeIndex = GetMemoryType(physicalDeviceMemoryProperties, pMemoryRequirements.memoryRequirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	vkAllocateMemory(device, &memoryAllocateInfo, 0, &pAccelerationStructure->memory);
	VkBindAccelerationStructureMemoryInfoNV bindInfo = { VK_STRUCTURE_TYPE_BIND_ACCELERATION_STRUCTURE_MEMORY_INFO_NV, 0, pAccelerationStructure->accelerationStructure, pAccelerationStructure->memory, 0, 0, 0};
	vkBindAccelerationStructureMemory(device, 1, &bindInfo);
	vkGetAccelerationStructureHandle(device, pAccelerationStructure->accelerationStructure, sizeof(uint64_t), &pAccelerationStructure->handle);
}

void Start(void* hwnd)
{
	const char* instanceExtensions[] = {"VK_KHR_surface", "VK_KHR_win32_surface"};
	VkInstanceCreateInfo instanceInfo = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, 0, 0, 0, 0, 0, 2, instanceExtensions };
	vkCreateInstance(&instanceInfo, 0, &instance);
	vkInit(instance);
	uint32_t numPhysicalDevices = 0;
	vkEnumeratePhysicalDevices(instance, &numPhysicalDevices, 0);
	VkPhysicalDevice* physicalDevices = (VkPhysicalDevice*)malloc(sizeof(VkPhysicalDevice) * numPhysicalDevices);
	vkEnumeratePhysicalDevices(instance, &numPhysicalDevices, physicalDevices);
	physicalDevice = physicalDevices[0];
	uint32_t queueFamilyPropertyCount;
	vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyPropertyCount, 0);
	VkQueueFamilyProperties* queueFamilyProperties = (VkQueueFamilyProperties*)malloc(sizeof(VkQueueFamilyProperties) * queueFamilyPropertyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyPropertyCount, queueFamilyProperties);
	VkDeviceQueueCreateInfo deviceQueueCreateInfo = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, 0, 0, 0, 1};
	const char* deviceExtensions[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_NV_RAY_TRACING_EXTENSION_NAME, VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME};
	VkDeviceCreateInfo deviceCreateInfo = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, 0, 0, 1, &deviceQueueCreateInfo, 0, 0, 3, deviceExtensions, 0};
	vkCreateDevice(physicalDevice, &deviceCreateInfo, 0, &device);
	vkGetDeviceQueue(device, 0, 0, &queue);
	VkPhysicalDeviceRayTracingPropertiesNV rtProps = (VkPhysicalDeviceRayTracingPropertiesNV){ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PROPERTIES_NV, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	VkPhysicalDeviceProperties2 deviceProperties2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &rtProps, 0 };
	vkGetPhysicalDeviceProperties2(physicalDevice, &deviceProperties2);
	VkWin32SurfaceCreateInfoKHR surfaceCreateInfo = { VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR, 0, 0, 0, (HWND)hwnd };
	vkCreateWin32SurfaceKHR(instance, &surfaceCreateInfo, 0, &surface);
	VkSwapchainCreateInfoKHR swapchainCreateInfo = { VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
	swapchainCreateInfo.flags = 0;
	swapchainCreateInfo.surface = surface;
	swapchainCreateInfo.minImageCount = 1;
	swapchainCreateInfo.imageFormat = VK_FORMAT_B8G8R8A8_UNORM;
	swapchainCreateInfo.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
	swapchainCreateInfo.imageExtent.width = ScreenWidth;
	swapchainCreateInfo.imageExtent.height = ScreenHeight;
	swapchainCreateInfo.imageArrayLayers = 1;
	swapchainCreateInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	swapchainCreateInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
	swapchainCreateInfo.queueFamilyIndexCount = 0;
	swapchainCreateInfo.pQueueFamilyIndices = 0;
	swapchainCreateInfo.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
	swapchainCreateInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	swapchainCreateInfo.presentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
	swapchainCreateInfo.clipped = VK_TRUE;
	swapchainCreateInfo.oldSwapchain = 0;
	vkCreateSwapchainKHR(device, &swapchainCreateInfo, 0, &swapchain);	
	vkGetSwapchainImagesKHR(device, swapchain, &imageCount, 0);
	swapchainImages = (VkImage*)malloc(sizeof(VkImage) * imageCount);
	vkGetSwapchainImagesKHR(device, swapchain, &imageCount, swapchainImages);
	swapchainImageViews = (VkImageView*)malloc(sizeof(VkImageView) * imageCount);
	for (uint32_t i = 0; i < imageCount; i++)
	{
		VkImageViewCreateInfo imageViewCreateInfo = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
		imageViewCreateInfo.format = VK_FORMAT_B8G8R8A8_UNORM;
		imageViewCreateInfo.subresourceRange = (VkImageSubresourceRange){ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
		imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		imageViewCreateInfo.image = swapchainImages[i];
		imageViewCreateInfo.flags = 0;
		imageViewCreateInfo.components = (VkComponentMapping){ VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B,VK_COMPONENT_SWIZZLE_A };
		vkCreateImageView(device, &imageViewCreateInfo, 0, &swapchainImageViews[i]);
	}
	VkFenceCreateInfo fenceCreateInfo = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, 0, VK_FENCE_CREATE_SIGNALED_BIT };
	waitForFrameFences = (VkFence*)malloc(sizeof(VkFence) * imageCount);
	for (int i=0; i<imageCount; i++)
		vkCreateFence(device, &fenceCreateInfo, 0, &waitForFrameFences[i]);
	VkCommandPoolCreateInfo commandPoolCreateInfo = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, 0, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, 0};
	vkCreateCommandPool(device, &commandPoolCreateInfo, 0, &commandPool);
	vkGetPhysicalDeviceMemoryProperties(physicalDevice, &physicalDeviceMemoryProperties);	
	commandBuffers = (VkCommandBuffer*)malloc(sizeof(VkCommandBuffer) * imageCount);
	VkCommandBufferAllocateInfo commandBufferAllocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, 0, commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, (uint32_t)(imageCount)};
	vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, commandBuffers);	
	VkSemaphoreCreateInfo semaphoreCreateInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, 0, 0 };
	vkCreateSemaphore(device, &semaphoreCreateInfo, 0, &semaphoreImageAcquired);
	vkCreateSemaphore(device, &semaphoreCreateInfo, 0, &semaphoreRenderFinished);
	struct ImageVulkan renderTexture;	
	renderTexture.type = VK_IMAGE_TYPE_2D;
	renderTexture.format = VK_FORMAT_B8G8R8A8_UNORM;
	renderTexture.extent = (VkExtent3D){ScreenWidth, ScreenHeight, 1};
	VkImageCreateInfo imageCreateInfo = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, 0, 0, renderTexture.type, renderTexture.format, renderTexture.extent, 1, 1, 1, 0, 8 | 1, 0, 0, 0, 0};
	vkCreateImage(device, &imageCreateInfo, 0, &renderTexture.image);
	VkMemoryRequirements memoryRequirements;
	vkGetImageMemoryRequirements(device, renderTexture.image, &memoryRequirements);
	VkMemoryAllocateInfo memoryAllocateInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
	memoryAllocateInfo.allocationSize = memoryRequirements.size;
	memoryAllocateInfo.memoryTypeIndex = GetMemoryType(physicalDeviceMemoryProperties, memoryRequirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	vkAllocateMemory(device, &memoryAllocateInfo, 0, &renderTexture.memory);
	vkBindImageMemory(device, renderTexture.image, renderTexture.memory, 0);
	VkImageSubresourceRange range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
	VkComponentMapping mapping = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B,VK_COMPONENT_SWIZZLE_A };
	VkImageViewCreateInfo imageViewCreateInfo = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, 0, 0, renderTexture.image, VK_IMAGE_VIEW_TYPE_2D, renderTexture.format, mapping, range};
	vkCreateImageView(device, &imageViewCreateInfo, 0, &renderTexture.view);
	float positions[][3] = {{1.0f, 1.0f, 0.0f}, {-1.0f, 1.0f, 0.0f}, {0.0f, -1.0f, 0.0f} };
	struct BufferVulkan positionsBuffer;
	struct BufferVulkan indicesBuffer;
	uint32_t indices[] = { 0, 1, 2 };
	LoadBuffer(device, sizeof(positions), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_RAY_TRACING_BIT_NV, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, positions , &positionsBuffer);
	LoadBuffer(device, sizeof(indices), VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_RAY_TRACING_BIT_NV, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, indices, &indicesBuffer);
	VkGeometryNV geometry = { VK_STRUCTURE_TYPE_GEOMETRY_NV };
	geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_NV;
	geometry.geometry.triangles = (VkGeometryTrianglesNV){ VK_STRUCTURE_TYPE_GEOMETRY_TRIANGLES_NV };
	geometry.geometry.triangles.vertexData = positionsBuffer.buffer;
	geometry.geometry.triangles.vertexOffset = 0;
	geometry.geometry.triangles.vertexCount = 3;
	geometry.geometry.triangles.vertexStride = sizeof(float) * 3;
	geometry.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
	geometry.geometry.triangles.indexData = indicesBuffer.buffer;
	geometry.geometry.triangles.indexOffset = 0;
	geometry.geometry.triangles.indexCount = 3;
	geometry.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32; 
	geometry.geometry.triangles.transformData = VK_NULL_HANDLE;
	geometry.geometry.triangles.transformOffset = 0;
	geometry.geometry.aabbs = (VkGeometryAABBNV){ VK_STRUCTURE_TYPE_GEOMETRY_AABB_NV };
	geometry.flags = VK_GEOMETRY_OPAQUE_BIT_NV;
	const float transform[12] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
	LoadAccelerationStructure(device, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_NV, 1, 0, &geometry , &blas);
	struct VkGeometryInstance item;
	memcpy(item.transform, transform, sizeof(transform));
	item.instanceId = 0;
	item.mask = 0xFF;
	item.instanceOffset = 0;
	item.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_CULL_DISABLE_BIT_NV;
	item.accelerationStructureHandle = blas.handle;
	struct BufferVulkan instancesBuffer;
	LoadBuffer(device,  sizeof(struct VkGeometryInstance), VK_BUFFER_USAGE_RAY_TRACING_BIT_NV, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &item, &instancesBuffer);
	LoadAccelerationStructure(device, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_NV,0, 1, 0, &tlas);
	VkAccelerationStructureMemoryRequirementsInfoNV memoryRequirementsInfo;
	memoryRequirementsInfo.pNext = 0;
	memoryRequirementsInfo.sType = (VkStructureType){ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_NV };
	memoryRequirementsInfo.type = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_BUILD_SCRATCH_NV;
	memoryRequirementsInfo.accelerationStructure = blas.accelerationStructure;
	VkMemoryRequirements2 memReqBlas;
	vkGetAccelerationStructureMemoryRequirements(device, &memoryRequirementsInfo, &memReqBlas);
	VkDeviceSize maxBlasSize = memReqBlas.memoryRequirements.size;
	VkMemoryRequirements2 memReqTlas;
	memoryRequirementsInfo.accelerationStructure = tlas.accelerationStructure;
	vkGetAccelerationStructureMemoryRequirements(device, &memoryRequirementsInfo, &memReqTlas);
	VkDeviceSize scratchBufferSize = (maxBlasSize > memReqTlas.memoryRequirements.size) ? maxBlasSize : memReqTlas.memoryRequirements.size;
	struct BufferVulkan scratchBuffer;
	LoadBuffer(device, scratchBufferSize, VK_BUFFER_USAGE_RAY_TRACING_BIT_NV, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT , 0, &scratchBuffer);
	VkCommandBufferAllocateInfo commandBufferAllocateInfo2 = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, 0, commandPool,  VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
	VkCommandBuffer cmdBuffer = VK_NULL_HANDLE;
	vkAllocateCommandBuffers(device, &commandBufferAllocateInfo2, &cmdBuffer);
	VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, 0,  VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, 0};
	vkBeginCommandBuffer(cmdBuffer, &beginInfo);	
	VkAccessFlags srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_NV | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_NV;
	VkAccessFlags dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_NV | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_NV;
	VkMemoryBarrier memoryBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, 0, srcAccessMask, dstAccessMask };
	blas.accelerationStructureInfo.instanceCount = 0;
	blas.accelerationStructureInfo.geometryCount = 1;
	blas.accelerationStructureInfo.pGeometries = &geometry;
	vkCmdBuildAccelerationStructure(cmdBuffer, &blas.accelerationStructureInfo, 0, 0, VK_FALSE, blas.accelerationStructure, 0, scratchBuffer.buffer, 0);
	vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV, 0, 1, &memoryBarrier, 0, 0, 0, 0);
	tlas.accelerationStructureInfo.instanceCount = 1;
	tlas.accelerationStructureInfo.geometryCount = 0;
	tlas.accelerationStructureInfo.pGeometries = 0;
	vkCmdBuildAccelerationStructure(cmdBuffer, &tlas.accelerationStructureInfo, instancesBuffer.buffer, 0, VK_FALSE, tlas.accelerationStructure, 0, scratchBuffer.buffer, 0);
	vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV, 0, 1, &memoryBarrier, 0, 0, 0, 0);
	vkEndCommandBuffer(cmdBuffer);
	VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO, 0, 0, 0, 0, 1, &cmdBuffer, 0, 0 };
	vkQueueSubmit(queue, 1, &submitInfo, 0);
	vkQueueWaitIdle(queue);
	vkFreeCommandBuffers(device, commandPool, 1, &cmdBuffer);
	ReleaseBuffer(device, scratchBuffer);	
	VkDescriptorSetLayoutBinding asLayoutBinding = {0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV, 1, VK_SHADER_STAGE_RAYGEN_BIT_NV, 0};
	VkDescriptorSetLayoutBinding outputImageLayoutBinding = {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_RAYGEN_BIT_NV};
	VkDescriptorSetLayoutBinding bindings[] = {asLayoutBinding, outputImageLayoutBinding};
	VkDescriptorSetLayoutCreateInfo set0LayoutInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, 0, 0, 2, bindings };
	vkCreateDescriptorSetLayout(device, &set0LayoutInfo, 0, &descriptorSetLayout);		
	VkShaderModule raygenShader, chitShader, missShader;
	VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, 0, 0, 1, &descriptorSetLayout, 0, 0 };
	vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, 0, &pipelineLayout);
	LoadShader(device, "raygen.spv", &raygenShader);
	LoadShader(device, "hit.spv", &chitShader);
	LoadShader(device, "miss.spv", &missShader);
	VkPipelineShaderStageCreateInfo raygenStage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, 0, 0, VK_SHADER_STAGE_RAYGEN_BIT_NV, raygenShader, "main", 0 };
	VkPipelineShaderStageCreateInfo chitStage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, 0, 0, VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV, chitShader, "main", 0 };
	VkPipelineShaderStageCreateInfo missStage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, 0, 0, VK_SHADER_STAGE_MISS_BIT_NV, missShader, "main" };
	VkRayTracingShaderGroupCreateInfoNV raygenGroup = { VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_NV };
	raygenGroup.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_NV;
	raygenGroup.generalShader = 0;
	raygenGroup.closestHitShader = VK_SHADER_UNUSED_NV;
	raygenGroup.anyHitShader = VK_SHADER_UNUSED_NV;
	raygenGroup.intersectionShader = VK_SHADER_UNUSED_NV;
	VkRayTracingShaderGroupCreateInfoNV chitGroup = { VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_NV };
	chitGroup.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_NV;
	chitGroup.generalShader = VK_SHADER_UNUSED_NV;
	chitGroup.closestHitShader = 1;
	chitGroup.anyHitShader = VK_SHADER_UNUSED_NV;
	chitGroup.intersectionShader = VK_SHADER_UNUSED_NV;
	VkRayTracingShaderGroupCreateInfoNV missGroup = { VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_NV };
	missGroup.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_NV;
	missGroup.generalShader = 2;
	missGroup.closestHitShader = VK_SHADER_UNUSED_NV;
	missGroup.anyHitShader = VK_SHADER_UNUSED_NV;
	missGroup.intersectionShader = VK_SHADER_UNUSED_NV;
	VkPipelineShaderStageCreateInfo stages[] = {raygenStage, chitStage, missStage};
	VkRayTracingShaderGroupCreateInfoNV groups[] = {raygenGroup, chitGroup, missGroup};
	VkRayTracingPipelineCreateInfoNV rayPipelineInfo = { VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_NV, 0, 0, 3, stages, 3, groups, 1, pipelineLayout, 0, 0 };
	vkCreateRayTracingPipelines(device, 0, 1, &rayPipelineInfo, 0, &rtPipeline);
	struct BufferVulkan sbtBuffer;
	LoadBuffer(device,  rtProps.shaderGroupHandleSize * 3, VK_BUFFER_USAGE_RAY_TRACING_BIT_NV, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT , 0, &sbtBuffer); 
	char* tmp = (char*)malloc(sbtBuffer.size);
	vkGetRayTracingShaderGroupHandles(device, rtPipeline, 0, 3, sbtBuffer.size, &tmp);
	void* sbtBufferMemory = 0;
	vkMapMemory(device, sbtBuffer.memory, 0, sbtBuffer.size, 0, &sbtBufferMemory);
	memcpy(sbtBufferMemory, &tmp, sbtBuffer.size);
	vkUnmapMemory(device, sbtBuffer.memory);
	VkDescriptorPoolSize poolSizes[] = {{ VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV, 1 }, { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 },{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 }};
	VkDescriptorPoolCreateInfo descPoolCreateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, 0, 0, 1, 3, poolSizes};
	vkCreateDescriptorPool(device, &descPoolCreateInfo, 0, &descriptorPool);
	VkDescriptorSetLayout setLayouts[] = {descriptorSetLayout};
	VkDescriptorSetAllocateInfo descSetAllocInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, 0,  descriptorPool, 1, setLayouts};
	vkAllocateDescriptorSets(device, &descSetAllocInfo, &descriptorSet);
	VkWriteDescriptorSetAccelerationStructureNV descAccelStructInfo = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_NV, 0, 1, &tlas.accelerationStructure };
	VkWriteDescriptorSet accelStructWrite = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, &descAccelStructInfo, descriptorSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV, 0, 0, 0};
	VkDescriptorImageInfo descOutputImageInfo = {0, renderTexture.view, 1};
	VkWriteDescriptorSet resImageWrite = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, 0, descriptorSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &descOutputImageInfo, 0, 0 };
	VkWriteDescriptorSet descriptorWrites[] = {accelStructWrite, resImageWrite};
	vkUpdateDescriptorSets(device, 2, descriptorWrites, 0, 0);
	VkCommandBufferBeginInfo commandBufferBeginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, 0, 0, 0 };
	VkImageSubresourceRange subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
	for (size_t i = 0; i < imageCount; i++)
	{
		const VkCommandBuffer cmdBuffer = commandBuffers[i];
		vkBeginCommandBuffer(cmdBuffer, &commandBufferBeginInfo);
		VkImageMemoryBarrier imageMemoryBarrier1 = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, 0, 0, VK_ACCESS_SHADER_WRITE_BIT, 0, 1, 0, 0, renderTexture.image, subresourceRange};
		vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, 0, 0, 0, 1, &imageMemoryBarrier1);		
		vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_NV, rtPipeline);
		vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_NV, pipelineLayout, 0, 1, &descriptorSet, 0, 0);
		uint32_t stride = rtProps.shaderGroupHandleSize;			
		vkCmdTraceRays(cmdBuffer, sbtBuffer.buffer, 0, sbtBuffer.buffer, stride * 2, stride, sbtBuffer.buffer, stride * 1, stride, 0, 0, 0, ScreenWidth, ScreenHeight, 1);
		VkImageMemoryBarrier imageMemoryBarrier2 = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, 0, 0, VK_ACCESS_TRANSFER_WRITE_BIT, 0, 7, 0, 0, swapchainImages[i], subresourceRange};
		vkCmdPipelineBarrier(cmdBuffer,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, 0, 0, 0, 1, &imageMemoryBarrier2);
		VkImageMemoryBarrier imageMemoryBarrier3 = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, 0, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, 1, 6, 0, 0, renderTexture.image, subresourceRange};
		vkCmdPipelineBarrier(cmdBuffer,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, 0, 0, 0, 1, &imageMemoryBarrier3);
		VkImageCopy copyRegion;
		copyRegion.srcSubresource = (VkImageSubresourceLayers){ VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
		copyRegion.srcOffset = (VkOffset3D){ 0, 0, 0 };
		copyRegion.dstSubresource = (VkImageSubresourceLayers){ VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
		copyRegion.dstOffset = (VkOffset3D){ 0, 0, 0 };
		copyRegion.extent = (VkExtent3D){ ScreenWidth, ScreenHeight, 1 };
		vkCmdCopyImage(cmdBuffer, renderTexture.image, 6, swapchainImages[i], 7, 1, &copyRegion);
		VkImageMemoryBarrier imageMemoryBarrier4 = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, 0, VK_ACCESS_TRANSFER_WRITE_BIT, 0, 7, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, 0, 0, swapchainImages[i], subresourceRange };
		vkCmdPipelineBarrier(cmdBuffer,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, 0, 0, 0, 1, &imageMemoryBarrier4);
		vkEndCommandBuffer(cmdBuffer);
	}	
}

static LRESULT CALLBACK WindowProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	if (uMsg == WM_CLOSE || uMsg == WM_DESTROY || (uMsg == WM_KEYDOWN && wParam == VK_ESCAPE))
	{
		PostQuitMessage(0); return 0;
	}
	else
	{
		return DefWindowProc(hWnd, uMsg, wParam, lParam);
	}
}

int WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
	int exit = 0;
	MSG msg;
	WNDCLASS win = { CS_OWNDC | CS_HREDRAW | CS_VREDRAW, WindowProc, 0, 0, 0, 0, 0, (HBRUSH)(COLOR_WINDOW + 1), 0, "Vulkan" };
	RegisterClass(&win);
	HWND hwnd = CreateWindowEx(0, win.lpszClassName, "Vulkan", WS_VISIBLE | WS_POPUP, 0, 0, ScreenWidth, ScreenHeight, 0, 0, 0, 0);
	Start(hwnd);
	while (!exit)
	{
		while (PeekMessage(&msg, 0, 0, 0, PM_REMOVE))
		{
			if (msg.message == WM_QUIT) exit = 1;
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
		uint32_t imageIndex = 0;
		vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, semaphoreImageAcquired, 0, &imageIndex);
		VkFence fence = waitForFrameFences[imageIndex];
		vkWaitForFences(device, 1, &fence, 1, UINT64_MAX);
		vkResetFences(device, 1, &fence);
		VkSubmitInfo submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO, 0, 1, &semaphoreImageAcquired, 0, 1, &commandBuffers[imageIndex], 1, &semaphoreRenderFinished };
		vkQueueSubmit(queue, 1, &submitInfo, fence);
		VkPresentInfoKHR presentInfo = { VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, 0, 1, &semaphoreRenderFinished, 1, &swapchain, &imageIndex, 0 };
		vkQueuePresentKHR(queue, &presentInfo);
	}
	return 0;
}