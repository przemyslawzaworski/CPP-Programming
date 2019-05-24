//cl.exe /EHsc /I"%VK_SDK_PATH%\include" pattern.c /link /LIBPATH:"%VK_SDK_PATH%\lib" user32.lib kernel32.lib gdi32.lib vulkan-1.lib
//Compile from GLSL to uint32_t: glslangValidator -V -x -o shader.u32 shader.frag

#include <windows.h>
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>
#include <stdlib.h>

const int WIDTH = 1280;
const int HEIGHT = 720;

static const uint32_t VertexShader[] =
{
	0x07230203,0x00010000,0x00080007,0x00000026,0x00000000,0x00020011,0x00000001,0x0006000b,
	0x00000001,0x4c534c47,0x6474732e,0x3035342e,0x00000000,0x0003000e,0x00000000,0x00000001,
	0x0007000f,0x00000000,0x00000004,0x6e69616d,0x00000000,0x00000017,0x0000001b,0x00030003,
	0x00000002,0x000001c2,0x00090004,0x415f4c47,0x735f4252,0x72617065,0x5f657461,0x64616873,
	0x6f5f7265,0x63656a62,0x00007374,0x00040005,0x00000004,0x6e69616d,0x00000000,0x00050005,
	0x0000000c,0x69736f70,0x6e6f6974,0x00000073,0x00060005,0x00000015,0x505f6c67,0x65567265,
	0x78657472,0x00000000,0x00060006,0x00000015,0x00000000,0x505f6c67,0x7469736f,0x006e6f69,
	0x00030005,0x00000017,0x00000000,0x00060005,0x0000001b,0x565f6c67,0x65747265,0x646e4978,
	0x00007865,0x00050048,0x00000015,0x00000000,0x0000000b,0x00000000,0x00030047,0x00000015,
	0x00000002,0x00040047,0x0000001b,0x0000000b,0x0000002a,0x00020013,0x00000002,0x00030021,
	0x00000003,0x00000002,0x00030016,0x00000006,0x00000020,0x00040017,0x00000007,0x00000006,
	0x00000002,0x00040015,0x00000008,0x00000020,0x00000000,0x0004002b,0x00000008,0x00000009,
	0x00000006,0x0004001c,0x0000000a,0x00000007,0x00000009,0x00040020,0x0000000b,0x00000006,
	0x0000000a,0x0004003b,0x0000000b,0x0000000c,0x00000006,0x0004002b,0x00000006,0x0000000d,
	0xbf800000,0x0005002c,0x00000007,0x0000000e,0x0000000d,0x0000000d,0x0004002b,0x00000006,
	0x0000000f,0x3f800000,0x0005002c,0x00000007,0x00000010,0x0000000f,0x0000000d,0x0005002c,
	0x00000007,0x00000011,0x0000000d,0x0000000f,0x0005002c,0x00000007,0x00000012,0x0000000f,
	0x0000000f,0x0009002c,0x0000000a,0x00000013,0x0000000e,0x00000010,0x00000011,0x00000012,
	0x00000011,0x00000010,0x00040017,0x00000014,0x00000006,0x00000004,0x0003001e,0x00000015,
	0x00000014,0x00040020,0x00000016,0x00000003,0x00000015,0x0004003b,0x00000016,0x00000017,
	0x00000003,0x00040015,0x00000018,0x00000020,0x00000001,0x0004002b,0x00000018,0x00000019,
	0x00000000,0x00040020,0x0000001a,0x00000001,0x00000018,0x0004003b,0x0000001a,0x0000001b,
	0x00000001,0x00040020,0x0000001d,0x00000006,0x00000007,0x0004002b,0x00000006,0x00000020,
	0x00000000,0x00040020,0x00000024,0x00000003,0x00000014,0x00050036,0x00000002,0x00000004,
	0x00000000,0x00000003,0x000200f8,0x00000005,0x0003003e,0x0000000c,0x00000013,0x0004003d,
	0x00000018,0x0000001c,0x0000001b,0x00050041,0x0000001d,0x0000001e,0x0000000c,0x0000001c,
	0x0004003d,0x00000007,0x0000001f,0x0000001e,0x00050051,0x00000006,0x00000021,0x0000001f,
	0x00000000,0x00050051,0x00000006,0x00000022,0x0000001f,0x00000001,0x00070050,0x00000014,
	0x00000023,0x00000021,0x00000022,0x00000020,0x0000000f,0x00050041,0x00000024,0x00000025,
	0x00000017,0x00000019,0x0003003e,0x00000025,0x00000023,0x000100fd,0x00010038
};

static const uint32_t FragmentShader[] =
{
	0x07230203,0x00010000,0x00080007,0x0000004d,0x00000000,0x00020011,0x00000001,0x0006000b,
	0x00000001,0x4c534c47,0x6474732e,0x3035342e,0x00000000,0x0003000e,0x00000000,0x00000001,
	0x0007000f,0x00000004,0x00000004,0x6e69616d,0x00000000,0x00000012,0x00000046,0x00030010,
	0x00000004,0x00000007,0x00030003,0x00000002,0x000001c2,0x00040005,0x00000004,0x6e69616d,
	0x00000000,0x00050005,0x00000009,0x73655269,0x74756c6f,0x006e6f69,0x00030005,0x0000000d,
	0x00007675,0x00060005,0x00000012,0x465f6c67,0x43676172,0x64726f6f,0x00000000,0x00030005,
	0x00000020,0x0000006b,0x00030005,0x0000002d,0x00000063,0x00050005,0x00000046,0x67617266,
	0x6f6c6f43,0x00000072,0x00040047,0x00000012,0x0000000b,0x0000000f,0x00040047,0x00000046,
	0x0000001e,0x00000000,0x00020013,0x00000002,0x00030021,0x00000003,0x00000002,0x00030016,
	0x00000006,0x00000020,0x00040017,0x00000007,0x00000006,0x00000002,0x00040020,0x00000008,
	0x00000007,0x00000007,0x0004002b,0x00000006,0x0000000a,0x44a00000,0x0004002b,0x00000006,
	0x0000000b,0x44340000,0x0005002c,0x00000007,0x0000000c,0x0000000a,0x0000000b,0x0004002b,
	0x00000006,0x0000000e,0x41400000,0x0004002b,0x00000006,0x0000000f,0x40000000,0x00040017,
	0x00000010,0x00000006,0x00000004,0x00040020,0x00000011,0x00000001,0x00000010,0x0004003b,
	0x00000011,0x00000012,0x00000001,0x00040015,0x00000019,0x00000020,0x00000000,0x0004002b,
	0x00000019,0x0000001a,0x00000001,0x00040020,0x0000001b,0x00000007,0x00000006,0x00040020,
	0x0000002c,0x00000007,0x00000010,0x0004002b,0x00000006,0x0000002e,0x3f800000,0x0004002b,
	0x00000019,0x0000002f,0x00000000,0x00040020,0x00000045,0x00000003,0x00000010,0x0004003b,
	0x00000045,0x00000046,0x00000003,0x00050036,0x00000002,0x00000004,0x00000000,0x00000003,
	0x000200f8,0x00000005,0x0004003b,0x00000008,0x00000009,0x00000007,0x0004003b,0x00000008,
	0x0000000d,0x00000007,0x0004003b,0x00000008,0x00000020,0x00000007,0x0004003b,0x0000002c,
	0x0000002d,0x00000007,0x0003003e,0x00000009,0x0000000c,0x0004003d,0x00000010,0x00000013,
	0x00000012,0x0007004f,0x00000007,0x00000014,0x00000013,0x00000013,0x00000000,0x00000001,
	0x0005008e,0x00000007,0x00000015,0x00000014,0x0000000f,0x0004003d,0x00000007,0x00000016,
	0x00000009,0x00050083,0x00000007,0x00000017,0x00000015,0x00000016,0x0005008e,0x00000007,
	0x00000018,0x00000017,0x0000000e,0x00050041,0x0000001b,0x0000001c,0x00000009,0x0000001a,
	0x0004003d,0x00000006,0x0000001d,0x0000001c,0x00050050,0x00000007,0x0000001e,0x0000001d,
	0x0000001d,0x00050088,0x00000007,0x0000001f,0x00000018,0x0000001e,0x0003003e,0x0000000d,
	0x0000001f,0x0004003d,0x00000007,0x00000021,0x0000000d,0x0006000c,0x00000007,0x00000022,
	0x00000001,0x0000000e,0x00000021,0x0004003d,0x00000007,0x00000023,0x0000000d,0x0006000c,
	0x00000007,0x00000024,0x00000001,0x0000000d,0x00000023,0x0007000c,0x00000007,0x00000025,
	0x00000001,0x00000025,0x00000022,0x00000024,0x0004003d,0x00000007,0x00000026,0x0000000d,
	0x0006000c,0x00000007,0x00000027,0x00000001,0x0000000e,0x00000026,0x0004003d,0x00000007,
	0x00000028,0x0000000d,0x0006000c,0x00000007,0x00000029,0x00000001,0x0000000d,0x00000028,
	0x0007000c,0x00000007,0x0000002a,0x00000001,0x00000028,0x00000027,0x00000029,0x00050081,
	0x00000007,0x0000002b,0x00000025,0x0000002a,0x0003003e,0x00000020,0x0000002b,0x00050041,
	0x0000001b,0x00000030,0x00000020,0x0000002f,0x0004003d,0x00000006,0x00000031,0x00000030,
	0x0006000c,0x00000006,0x00000032,0x00000001,0x0000000e,0x00000031,0x00050083,0x00000006,
	0x00000033,0x0000002e,0x00000032,0x00050041,0x0000001b,0x00000034,0x00000020,0x0000001a,
	0x0004003d,0x00000006,0x00000035,0x00000034,0x00050041,0x0000001b,0x00000036,0x00000020,
	0x0000002f,0x0004003d,0x00000006,0x00000037,0x00000036,0x00050041,0x0000001b,0x00000038,
	0x00000020,0x0000001a,0x0004003d,0x00000006,0x00000039,0x00000038,0x0007000c,0x00000006,
	0x0000003a,0x00000001,0x00000028,0x00000037,0x00000039,0x0006000c,0x00000006,0x0000003b,
	0x00000001,0x0000000d,0x0000003a,0x00050085,0x00000006,0x0000003c,0x00000035,0x0000003b,
	0x00050083,0x00000006,0x0000003d,0x0000002e,0x0000003c,0x00050041,0x0000001b,0x0000003e,
	0x00000020,0x0000002f,0x0004003d,0x00000006,0x0000003f,0x0000003e,0x00050041,0x0000001b,
	0x00000040,0x00000020,0x0000001a,0x0004003d,0x00000006,0x00000041,0x00000040,0x00050085,
	0x00000006,0x00000042,0x0000003f,0x00000041,0x00050083,0x00000006,0x00000043,0x0000002e,
	0x00000042,0x00070050,0x00000010,0x00000044,0x00000033,0x0000003d,0x00000043,0x0000002e,
	0x0003003e,0x0000002d,0x00000044,0x0004003d,0x00000010,0x00000047,0x0000002d,0x0004003d,
	0x00000010,0x00000048,0x0000002d,0x0009004f,0x00000010,0x00000049,0x00000048,0x00000048,
	0x00000001,0x00000002,0x00000000,0x00000003,0x0006000c,0x00000006,0x0000004a,0x00000001,
	0x00000042,0x00000049,0x0006000c,0x00000006,0x0000004b,0x00000001,0x0000000d,0x0000004a,
	0x0005008e,0x00000010,0x0000004c,0x00000047,0x0000004b,0x0003003e,0x00000046,0x0000004c,
	0x000100fd,0x00010038
};

static LRESULT CALLBACK WindowProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	if (uMsg == WM_KEYUP && wParam == VK_ESCAPE)
	{
		PostQuitMessage(0);
		return 0;
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
	WNDCLASSEX WndClassEx = { 0 };
	WndClassEx.cbSize = sizeof(WndClassEx);
	WndClassEx.lpfnWndProc = WindowProc;
	WndClassEx.lpszClassName = "Vulkan Demo";
	RegisterClassEx(&WndClassEx);
	HWND hwnd = CreateWindowEx(WS_EX_LEFT, WndClassEx.lpszClassName, NULL, WS_POPUP, 0, 0, WIDTH, HEIGHT, 0, 0, NULL, NULL);
	ShowWindow(hwnd, SW_SHOW);
	const char* device_extensions[] = { "VK_KHR_swapchain" };
	uint32_t deviceCount = 0, imageCount = 1;
	VkInstance instance;
	VkSurfaceKHR surface;
	VkDevice device;
	VkQueue graphicsQueue, presentQueue = NULL;
	VkSwapchainKHR swapChain;
	VkFormat swapChainImageFormat;
	VkRenderPass renderPass;
	VkPipelineLayout pipelineLayout;
	VkPipeline graphicsPipeline;
	VkCommandPool commandPool;
	VkSemaphore imageAvailableSemaphore, renderFinishedSemaphore;
	const char* glfwExtensions[] = {"VK_KHR_surface", "VK_KHR_win32_surface"};
	VkInstanceCreateInfo instanceInfo = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, 0, 0, 0, 0, 0, 2, glfwExtensions };
	vkCreateInstance(&instanceInfo, 0, &instance);
	VkWin32SurfaceCreateInfoKHR win32surfaceCreateInfo = { VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR ,0,0,0, hwnd };
	vkCreateWin32SurfaceKHR(instance, &win32surfaceCreateInfo, 0, &surface);
	vkEnumeratePhysicalDevices(instance, &deviceCount, 0);
	VkPhysicalDevice* devices = (VkPhysicalDevice*)malloc(sizeof(VkPhysicalDevice) * deviceCount);
	vkEnumeratePhysicalDevices(instance, &deviceCount, devices);
	VkPhysicalDevice physicalDevice = devices[0];
	VkDeviceQueueCreateInfo queueCreateInfo = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO ,0, 0, 0 ,1 };
	VkDeviceCreateInfo deviceInfo = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,0,0,1,&queueCreateInfo,0,0,1,device_extensions,0 };
	vkCreateDevice(physicalDevice, &deviceInfo, 0, &device);
	vkGetDeviceQueue(device, 0, 0, &graphicsQueue);
	vkGetDeviceQueue(device, 0, 0, &presentQueue);
	VkSurfaceFormatKHR surfaceFormat = { VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
	VkExtent2D extent = { WIDTH, HEIGHT };
	VkSwapchainCreateInfoKHR swapInfo = { VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,0,0,surface,imageCount,surfaceFormat.format ,surfaceFormat.colorSpace,extent,1,16,VK_SHARING_MODE_EXCLUSIVE,0,0,VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,VK_PRESENT_MODE_IMMEDIATE_KHR,1,0 };
	vkCreateSwapchainKHR(device, &swapInfo, 0, &swapChain);
	vkGetSwapchainImagesKHR(device, swapChain, &imageCount, 0);
	VkImage* swapChainImages = (VkImage*)malloc(sizeof(VkImage) * 2);
	vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages);
	swapChainImageFormat = surfaceFormat.format;
	VkExtent2D swapChainExtent = extent;
	VkImageView* swapChainImageViews = (VkImageView*)malloc(sizeof(VkImageView) * 2);
	for (size_t i = 0; i < 2; i++)
	{
		VkComponentMapping swizzle = { VK_COMPONENT_SWIZZLE_IDENTITY ,VK_COMPONENT_SWIZZLE_IDENTITY ,VK_COMPONENT_SWIZZLE_IDENTITY ,VK_COMPONENT_SWIZZLE_IDENTITY };
		VkImageSubresourceRange subresource = { VK_IMAGE_ASPECT_COLOR_BIT ,0,1,0,1 };
		VkImageViewCreateInfo imageInfo = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO ,0,0,swapChainImages[i],VK_IMAGE_VIEW_TYPE_2D,swapChainImageFormat,swizzle, subresource };
		vkCreateImageView(device, &imageInfo, 0, &swapChainImageViews[i]);
	}
	VkAttachmentDescription colorAttachment = { 0,swapChainImageFormat,VK_SAMPLE_COUNT_1_BIT ,VK_ATTACHMENT_LOAD_OP_CLEAR ,VK_ATTACHMENT_STORE_OP_STORE ,VK_ATTACHMENT_LOAD_OP_DONT_CARE ,VK_ATTACHMENT_STORE_OP_DONT_CARE ,VK_IMAGE_LAYOUT_UNDEFINED ,VK_IMAGE_LAYOUT_PRESENT_SRC_KHR };
	VkAttachmentReference colorAttachmentRef = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
	VkSubpassDescription subpass = { 0,VK_PIPELINE_BIND_POINT_GRAPHICS,0,0,1,&colorAttachmentRef,0,0,0,0 };
	VkSubpassDependency dependency = { VK_SUBPASS_EXTERNAL ,0,1024,1024 ,0,128 | 256 ,0 };
	VkRenderPassCreateInfo renderPassInfo = { VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO ,0,0,1,&colorAttachment ,1,&subpass ,1, &dependency };
	vkCreateRenderPass(device, &renderPassInfo, 0, &renderPass);
	VkShaderModule vertShaderModule, fragShaderModule;
	VkShaderModuleCreateInfo vsInfo = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO ,0,0,sizeof(VertexShader), VertexShader };
	vkCreateShaderModule(device, &vsInfo, 0, &vertShaderModule);
	VkShaderModuleCreateInfo fsInfo = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO ,0,0,sizeof(FragmentShader), FragmentShader };
	vkCreateShaderModule(device, &fsInfo, 0, &fragShaderModule);
	VkPipelineShaderStageCreateInfo vertShaderStageInfo = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO ,0,0,VK_SHADER_STAGE_VERTEX_BIT ,vertShaderModule ,"main" ,0 };
	VkPipelineShaderStageCreateInfo fragShaderStageInfo = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO ,0,0,VK_SHADER_STAGE_FRAGMENT_BIT,fragShaderModule, "main" ,0 };
	VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };
	VkPipelineVertexInputStateCreateInfo vertexInputInfo = { VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO ,0,0,0,0,0,0 };
	VkPipelineInputAssemblyStateCreateInfo inputAssembly = { VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO ,0,0,VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST ,0 };
	VkViewport viewport = { 0.0f,0.0f, WIDTH ,HEIGHT ,0.0f,1.0f };
	VkRect2D scissor = { { 0, 0 } ,swapChainExtent };
	VkPipelineViewportStateCreateInfo viewportState = { VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO ,0,0,1,&viewport ,1,&scissor };
	VkPipelineRasterizationStateCreateInfo rasterizer = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO ,0,0,0,0, VK_POLYGON_MODE_FILL ,VK_CULL_MODE_BACK_BIT ,VK_FRONT_FACE_CLOCKWISE,0,1.0f };
	VkPipelineMultisampleStateCreateInfo multisampling = { (VkStructureType)24 ,0,0,VK_SAMPLE_COUNT_1_BIT ,0,0,0,0,0 };
	VkPipelineColorBlendAttachmentState colorBlendAttachment = { 0,VK_BLEND_FACTOR_ZERO,VK_BLEND_FACTOR_ZERO,VK_BLEND_OP_ADD,VK_BLEND_FACTOR_ZERO,VK_BLEND_FACTOR_ZERO,VK_BLEND_OP_ADD, 1 | 2 | 4 | 8 };
	VkPipelineColorBlendStateCreateInfo colorBlending = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO ,0,0,0,VK_LOGIC_OP_COPY ,1,&colorBlendAttachment ,0 };
	VkPipelineLayoutCreateInfo pipelineLayoutInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO ,0,0,0,0,0,0 };
	vkCreatePipelineLayout(device, &pipelineLayoutInfo, 0, &pipelineLayout);
	VkGraphicsPipelineCreateInfo pipelineInfo = { (VkStructureType)28 ,0,0,2,shaderStages ,&vertexInputInfo ,&inputAssembly ,0,&viewportState, &rasterizer,&multisampling,0,&colorBlending,0,pipelineLayout, renderPass ,0,0 };
	vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, 0, &graphicsPipeline);
	vkDestroyShaderModule(device, fragShaderModule, 0);
	vkDestroyShaderModule(device, vertShaderModule, 0);
	VkFramebuffer* swapChainFramebuffers = (VkFramebuffer*)malloc(sizeof(VkFramebuffer) * 2);
	for (int i = 0; i < 2; i++)
	{
		VkImageView attachments[] = { swapChainImageViews[i] };
		VkFramebufferCreateInfo framebufferInfo = { VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO ,0,0,renderPass ,1,attachments,swapChainExtent.width,swapChainExtent.height,1 };
		vkCreateFramebuffer(device, &framebufferInfo, 0, &swapChainFramebuffers[i]);
	}
	VkCommandPoolCreateInfo poolInfo = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO , 0, 0, 0 };
	vkCreateCommandPool(device, &poolInfo, 0, &commandPool);
	VkCommandBuffer* commandBuffers = (VkCommandBuffer*)malloc(sizeof(VkCommandBuffer) * 2);
	VkCommandBufferAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, 0, commandPool ,VK_COMMAND_BUFFER_LEVEL_PRIMARY, 2 };
	vkAllocateCommandBuffers(device, &allocInfo, commandBuffers);
	for (int i = 0; i < 2; i++)
	{
		VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,0,VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, 0 };
		vkBeginCommandBuffer(commandBuffers[i], &beginInfo);
		VkRect2D rect = { { 0, 0 } ,swapChainExtent };
		VkClearValue clearColor = { 0.0f, 0.0f, 0.0f, 1.0f };
		VkRenderPassBeginInfo renderPassInfo = { VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO ,0,renderPass ,swapChainFramebuffers[i] , rect ,1,&clearColor };
		vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
		vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
		vkCmdDraw(commandBuffers[i], 6, 2, 0, 0);
		vkCmdEndRenderPass(commandBuffers[i]);
		vkEndCommandBuffer(commandBuffers[i]);
	}
	VkSemaphoreCreateInfo semaphoreInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO ,0,0 };
	vkCreateSemaphore(device, &semaphoreInfo, 0, &imageAvailableSemaphore);
	vkCreateSemaphore(device, &semaphoreInfo, 0, &renderFinishedSemaphore);
	while (!exit)
	{
		while(PeekMessage(&msg, 0, 0, 0, PM_REMOVE) )
		{
			if( msg.message==WM_QUIT ) exit = 1;
			TranslateMessage( &msg );
			DispatchMessage( &msg );
		}		
		uint32_t imageIndex;
		vkAcquireNextImageKHR(device, swapChain, (uint64_t) 1e16, imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);
		VkSemaphore waitSemaphores[] = { imageAvailableSemaphore };
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		VkSemaphore signalSemaphores[] = { renderFinishedSemaphore };
		VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO,0,1,waitSemaphores,waitStages,1,&commandBuffers[imageIndex],1 ,signalSemaphores };
		vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
		VkSwapchainKHR swapChains[] = { swapChain };
		VkPresentInfoKHR presentInfo = { VK_STRUCTURE_TYPE_PRESENT_INFO_KHR ,0,1,signalSemaphores ,1,swapChains ,&imageIndex ,0 };
		vkQueuePresentKHR(presentQueue, &presentInfo);
		vkQueueWaitIdle(presentQueue);
	}
	vkDeviceWaitIdle(device);
	vkDestroySemaphore(device, renderFinishedSemaphore, 0);
	vkDestroySemaphore(device, imageAvailableSemaphore, 0);
	vkDestroyCommandPool(device, commandPool, 0);
	for (int i = 0; i < 2; i++) vkDestroyFramebuffer(device, swapChainFramebuffers[i], 0);	
	vkDestroyPipeline(device, graphicsPipeline, 0);
	vkDestroyPipelineLayout(device, pipelineLayout, 0);
	vkDestroyRenderPass(device, renderPass, 0);
	for (int i = 0; i < 2; i++) vkDestroyImageView(device, swapChainImageViews[i], 0);
	vkDestroySwapchainKHR(device, swapChain, 0);
	vkDestroyDevice(device, 0);
	vkDestroySurfaceKHR(instance, surface, 0);
	vkDestroyInstance(instance, 0);
	return 0;
}