// cl.exe /EHsc /I"%VK_SDK_PATH%\include" demo.cpp /link /LIBPATH:"%VK_SDK_PATH%\lib" user32.lib kernel32.lib gdi32.lib vulkan-1.lib
#include <windows.h>
#include <stdio.h>
#include <vulkan/vulkan.h>

float ScreenWidth = 1280.0f;
float ScreenHeight = 720.0f;

struct ImageDescriptor
{
	VkImage image;
	VkDeviceMemory memory;
	VkImageView view;
	uint32_t width;
	uint32_t height;
};

void CreateImage(VkDevice device, VkPhysicalDevice pDevice, VkFormat format, uint32_t width, uint32_t height, ImageDescriptor& out)
{
	VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
	VkExtent3D extent = {width, height, 1};
	VkImageCreateInfo imageInfo = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, 0, 0, VK_IMAGE_TYPE_2D, format, extent, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_LINEAR, usage, VK_SHARING_MODE_EXCLUSIVE, 0, 0, VK_IMAGE_LAYOUT_UNDEFINED};
	vkCreateImage(device, &imageInfo, 0, &out.image);
	VkMemoryRequirements memoryRequirements;
	vkGetImageMemoryRequirements(device, out.image, &memoryRequirements);
	uint32_t index = 0u;
	VkPhysicalDeviceMemoryProperties properties{};
	vkGetPhysicalDeviceMemoryProperties(pDevice, &properties);
	for (uint32_t i = 0; i < properties.memoryTypeCount; i++) 
	{
		if ((memoryRequirements.memoryTypeBits & (1 << i)) && (properties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) 
		{
			index = i;
		}
	}
	VkMemoryAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, 0, memoryRequirements.size, index};
	vkAllocateMemory(device, &allocInfo, 0, &out.memory);
	vkBindImageMemory(device, out.image, out.memory, 0); 
	VkComponentMapping components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
	VkImageViewCreateInfo createInfo = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, 0, 0, out.image, VK_IMAGE_VIEW_TYPE_2D, format, components, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
	vkCreateImageView(device, &createInfo, 0, &out.view);
}

void DestroyImage(VkDevice device, struct ImageDescriptor* img) 
{
	vkDestroyImage(device, img->image, 0);
	vkDestroyImageView(device, img->view, 0);
	vkFreeMemory(device, img->memory, 0);
}

static LRESULT CALLBACK WindowProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	if (uMsg==WM_CLOSE || uMsg==WM_DESTROY || (uMsg==WM_KEYDOWN && wParam==VK_ESCAPE))
	{
		PostQuitMessage(0); return 0;
	}
	else
	{
		return DefWindowProc(hWnd, uMsg, wParam, lParam);
	}
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
	int exit = 0;
	MSG msg;
	WNDCLASS win = {CS_OWNDC|CS_HREDRAW|CS_VREDRAW, WindowProc, 0, 0, 0, 0, 0, (HBRUSH)(COLOR_WINDOW+1), 0, "Demo"};
	RegisterClass(&win);
	HWND hwnd = CreateWindowEx(0, win.lpszClassName, "Vulkan", WS_VISIBLE|WS_OVERLAPPEDWINDOW, 0, 0, ScreenWidth, ScreenHeight, 0, 0, 0, 0);
	HDC hdc = GetDC(hwnd);
	BITMAPINFO bmi = {{sizeof(BITMAPINFOHEADER), (long)ScreenWidth, (long)ScreenHeight, 1, 32, BI_RGB, 0, 0, 0, 0, 0}, {0, 0, 0, 0}};	
	VkInstance instance;
	const char** extensions = nullptr;
	VkApplicationInfo appInfo = {VK_STRUCTURE_TYPE_APPLICATION_INFO, 0, "Vulkan", 0, 0, 0, VK_API_VERSION_1_1};
	VkInstanceCreateInfo createInfo = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, 0, 0, &appInfo, 0, 0, 0, extensions};
	vkCreateInstance(&createInfo, 0, &instance);
	VkPhysicalDevice physicalDevice;
	uint32_t queueFamilyIndex = 0;
	uint32_t deviceCount = 0;
	vkEnumeratePhysicalDevices(instance, &deviceCount, 0);
	VkPhysicalDevice* devices = (VkPhysicalDevice*)malloc(sizeof(VkPhysicalDevice) * deviceCount);
	vkEnumeratePhysicalDevices(instance, &deviceCount, devices);
	for (uint32_t i = 0; i < deviceCount; i++) 
	{
		uint32_t count = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &count, 0);
		VkQueueFamilyProperties* queueFamilies = (VkQueueFamilyProperties*)malloc(sizeof(VkQueueFamilyProperties) * count);
		vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &count, queueFamilies);
		bool hasId = false;
		for (uint32_t j = 0; j < count; j++) 
		{
			if (queueFamilies[j].queueFlags & VK_QUEUE_GRAPHICS_BIT) 
			{
				hasId = true;
				queueFamilyIndex = j;
				break;
			}
		}
		free(queueFamilies);
		if (hasId) 
		{
			physicalDevice = devices[i];
			break;
		}
	}
	VkDevice device;
	float priority = 1.0f;
	VkDeviceQueueCreateInfo queueCreateInfo = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, 0, 0, 0, 1, &priority};
	VkDeviceCreateInfo deviceInfo = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, 0, 0, 1, &queueCreateInfo, 0, 0, 0, 0, 0};
	vkCreateDevice(physicalDevice, &deviceInfo, 0, &device);
	VkQueue queue;
	vkGetDeviceQueue(device, 0, 0, &queue);
	ImageDescriptor reader;
	CreateImage(device, physicalDevice, VK_FORMAT_R8G8B8A8_UNORM, 256, 256, reader);
	reader.width = reader.height = 256;
	uint32_t* pixels;
	vkMapMemory(device, reader.memory, 0, VK_WHOLE_SIZE, 0, (void**)&pixels);
	for (int x = 0; x < 256; x++) 
	{
		for (int y = 0; y < 256; y++) 
		{
			uint32_t hash = x * 0x1f1f1f1f + y * 0x3f3f3f3f;
			hash ^= hash >> 16;
			hash *= 0x85ebca6b;
			hash ^= hash >> 13;
			hash *= 0xc2b2ae35;
			hash ^= hash >> 16;
			uint8_t red = (hash & 0xFF);
			uint8_t green = ((hash >> 8) & 0xFF);
			uint8_t blue = ((hash >> 16) & 0xFF);
			uint8_t alpha = 255;
			pixels[x * 256 + y] = red | (green << 8u) | (blue << 16u) | (alpha << 24u);
		}
	}
	VkMappedMemoryRange range = {VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, 0, reader.memory, 0, VK_WHOLE_SIZE};
	vkFlushMappedMemoryRanges(device, 1, &range);
	vkUnmapMemory(device, reader.memory);
	ImageDescriptor writer;
	CreateImage(device, physicalDevice, VK_FORMAT_R8G8B8A8_UNORM, ScreenWidth, ScreenHeight, writer);
	writer.width = ScreenWidth;
	writer.height = ScreenHeight;	
	VkShaderModule shaderModule;
	FILE* file = fopen("demo.spv", "rb");
	fseek(file, 0, SEEK_END);
	long fileSize = ftell(file);
	fseek(file, 0, SEEK_SET);
	size_t size = (size_t) fileSize;
	uint32_t* shaderCode = (uint32_t*)malloc(size);
	size_t bytesRead = fread(shaderCode, 1, size, file);
	fclose(file);	
	VkShaderModuleCreateInfo shaderCreateInfo = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, 0, 0, size, shaderCode};
	vkCreateShaderModule(device, &shaderCreateInfo, 0, &shaderModule);
	VkDescriptorSetLayout descriptorSetLayout;
	VkDescriptorSetLayoutBinding layoutBindings[] = 
	{
		{0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0 },
		{1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0 },
	};
	VkDescriptorSetLayoutCreateInfo layoutCreateInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, 0, 0, sizeof(layoutBindings) / sizeof(layoutBindings[0]), layoutBindings};
	vkCreateDescriptorSetLayout(device, &layoutCreateInfo, 0, &descriptorSetLayout);
	VkPipelineLayout computePipelineLayout;
	VkPushConstantRange pushConstantRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int32_t) * 2};
	VkPipelineLayoutCreateInfo pipelineLayoutInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, 0, 0, 1, &descriptorSetLayout, 1, &pushConstantRange};
	vkCreatePipelineLayout(device, &pipelineLayoutInfo, 0, &computePipelineLayout);
	VkPipeline computePipeline;
	VkPipelineShaderStageCreateInfo computeStageInfo = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, 0, 0, VK_SHADER_STAGE_COMPUTE_BIT, shaderModule, "main", NULL};
	VkComputePipelineCreateInfo pPipelineInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, 0, 0, computeStageInfo, computePipelineLayout, VK_NULL_HANDLE, 0};
	vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pPipelineInfo, 0, &computePipeline);
	VkDescriptorPool descriptorPool;
	VkDescriptorPoolSize descriptorPoolSizes[] = {{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2}};
	VkDescriptorPoolCreateInfo poolCreateInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, 0, 0, 1, sizeof(descriptorPoolSizes) / sizeof(descriptorPoolSizes[0]), descriptorPoolSizes};
	vkCreateDescriptorPool(device, &poolCreateInfo, 0, &descriptorPool);
	VkDescriptorSet descriptorSet;
	VkDescriptorSetAllocateInfo setAllocateInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, 0, descriptorPool, 1, &descriptorSetLayout};
	vkAllocateDescriptorSets(device, &setAllocateInfo, &descriptorSet);
	VkDescriptorImageInfo srcInfo = {VK_NULL_HANDLE, reader.view, VK_IMAGE_LAYOUT_GENERAL};
	VkDescriptorImageInfo dstInfo = {VK_NULL_HANDLE, writer.view, VK_IMAGE_LAYOUT_GENERAL};
	VkWriteDescriptorSet writeDescriptorSet[] =
	{
		{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, 0, descriptorSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &srcInfo, 0, 0},
		{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, 0, descriptorSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &dstInfo, 0, 0},
	};
	vkUpdateDescriptorSets(device, 2, writeDescriptorSet, 0, 0);
	VkCommandPool commandPool;
	VkCommandPoolCreateInfo poolInfo = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, 0, 0, queueFamilyIndex};
	vkCreateCommandPool(device, &poolInfo, 0, &commandPool);
	VkCommandBuffer commandBuffer;
	VkCommandBufferAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, 0, commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
	vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);
	VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, 0, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, 0};
	VkFence fence;
	VkFenceCreateInfo fenceInfo = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, 0, 0};
	vkCreateFence(device, &fenceInfo, 0, &fence);	
	uint8_t* data;
	vkMapMemory(device, writer.memory, 0, VK_WHOLE_SIZE, 0, (void**)&data);
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &descriptorSet, 0, 0);	
	while (!exit)
	{
		while(PeekMessage(&msg, 0, 0, 0, PM_REMOVE))
		{
			if( msg.message==WM_QUIT ) exit = 1;
			TranslateMessage( &msg );
			DispatchMessage( &msg );
		}
		vkBeginCommandBuffer(commandBuffer, &beginInfo);
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
		float constants[] = {(float)GetTickCount() * 0.001f};
		vkCmdPushConstants(commandBuffer, computePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(float), constants);
		vkCmdDispatch(commandBuffer, ScreenWidth / 16, ScreenHeight / 16, 1);
		vkEndCommandBuffer(commandBuffer);
		VkSubmitInfo submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO, 0, 0, 0, 0, 1, &commandBuffer, 0, 0};
		vkQueueSubmit(queue, 1, &submitInfo, fence);
		vkWaitForFences(device, 1, &fence, 1u, -1);
		StretchDIBits(hdc, 0, 0, ScreenWidth, ScreenHeight, 0, 0, ScreenWidth, ScreenHeight, data, &bmi, DIB_RGB_COLORS, SRCCOPY);
	}
	vkUnmapMemory(device, writer.memory);
	vkDestroyFence(device, fence, 0);
	vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
	vkDestroyCommandPool(device, commandPool, 0);
	DestroyImage(device, &reader);
	DestroyImage(device, &writer);
	vkDestroyPipeline(device, computePipeline, 0);
	vkDestroyPipelineLayout(device, computePipelineLayout, 0);
	vkDestroyDescriptorPool(device, descriptorPool, 0);
	vkDestroyDescriptorSetLayout(device, descriptorSetLayout, 0);
	vkDestroyShaderModule(device, shaderModule, 0);
	vkDestroyDevice(device, 0);
	vkDestroyInstance(instance, 0);
	free(devices);
	free(shaderCode);
	return 0;
}