#include "application.hpp"

#include <imgui.h>

#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>

#include <filesystem>
#include <windows.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace application {

    namespace {

        struct Vertex {
            glm::vec3 position;
            glm::vec3 color;
        };

        struct ObjectData {
            glm::mat4 mvp;
            glm::vec4 color;
        };

        VkBuffer vertex_buffer = VK_NULL_HANDLE;
        VmaAllocation vertex_buffer_allocation = VK_NULL_HANDLE;

        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        VkPipeline graphics_pipeline = VK_NULL_HANDLE;

        VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
        VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;

        std::array<VkDescriptorSet, 2> descriptor_sets{};

        std::array<VkBuffer, 2> uniform_buffers{};
        std::array<VmaAllocation, 2> uniform_buffer_allocations{};

        glm::vec3 object_position = glm::vec3(0.0f);
        glm::vec3 object_rotation = glm::vec3(0.0f);
        glm::vec3 object_scale = glm::vec3(1.0f);
        glm::vec3 object_color(0.2f, 0.5f, 1.0f);

        glm::vec3 second_object_position = glm::vec3(0.0f);
        glm::vec3 second_object_rotation = glm::vec3(0.0f);

        bool use_perspective = true;

        bool animation_enabled = true;
        float animation_speed = 1.0f;
        float trajectory_radius = 1.5f;

        std::vector<Vertex> vertices;

        std::vector<char> readFile(const std::filesystem::path& filename) {
            std::ifstream file(filename, std::ios::ate | std::ios::binary);

            if (!file) {
                return {};
            }

            const size_t file_size = static_cast<size_t>(file.tellg());

            std::vector<char> buffer(file_size);

            file.seekg(0);
            file.read(buffer.data(), static_cast<std::streamsize>(file_size));

            return buffer;
        }

        VkShaderModule createShaderModule(const std::vector<char>& code) {
            if (code.empty() || code.size() % 4 != 0) {
                return VK_NULL_HANDLE;
            }

            const VkShaderModuleCreateInfo create_info = {
                .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                .codeSize = code.size(),
                .pCode = reinterpret_cast<const uint32_t*>(code.data())
            };

            VkShaderModule shader_module = VK_NULL_HANDLE;

            if (vkCreateShaderModule(
                graphics::internal::context.device,
                &create_info,
                nullptr,
                &shader_module) != VK_SUCCESS) {
                return VK_NULL_HANDLE;
            }

            return shader_module;
        }

        void createCone() {
            constexpr uint32_t segments = 32;

            constexpr float radius = 1.0f;
            constexpr float half_height = 1.0f;

            const glm::vec3 apex(0.0f, half_height, 0.0f);
            const glm::vec3 center(0.0f, -half_height, 0.0f);

            vertices.clear();
            vertices.reserve(segments * 6);

            constexpr float pi = 3.14159265358979323846f;

            auto getVertexColor = [](const glm::vec3& position) {
                return glm::vec3(
                    0.5f + 0.5f * position.x,
                    0.5f + 0.5f * position.y,
                    0.5f + 0.5f * position.z
                );
            };

            for (uint32_t i = 0; i < segments; ++i) {
                const float angle1 =
                    2.0f * pi * static_cast<float>(i) / static_cast<float>(segments);

                const float angle2 =
                    2.0f * pi * static_cast<float>(i + 1) / static_cast<float>(segments);

                const glm::vec3 p1(
                    radius * std::cos(angle1),
                    -half_height,
                    radius * std::sin(angle1));

                const glm::vec3 p2(
                    radius * std::cos(angle2),
                    -half_height,
                    radius * std::sin(angle2));

                // Боковая поверхность.
                vertices.push_back({ apex, getVertexColor(apex) });
                vertices.push_back({ p1, getVertexColor(p1) });
                vertices.push_back({ p2, getVertexColor(p2) });

                // Основание.
                vertices.push_back({ center, getVertexColor(center) });
                vertices.push_back({ p2, getVertexColor(p2) });
                vertices.push_back({ p1, getVertexColor(p1) });
            }
        }

        bool createVertexBuffer() {
            const VkDeviceSize buffer_size =
                sizeof(Vertex) * vertices.size();

            const VkBufferCreateInfo buffer_create_info = {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .size = buffer_size,
                .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE
            };

            const VmaAllocationCreateInfo allocation_create_info = {
                .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                .usage = VMA_MEMORY_USAGE_AUTO
            };

            if (vmaCreateBuffer(
                graphics::internal::context.allocator,
                &buffer_create_info,
                &allocation_create_info,
                &vertex_buffer,
                &vertex_buffer_allocation,
                nullptr) != VK_SUCCESS) {
                std::cerr << "Failed to create vertex buffer\n";
                return false;
            }

            void* mapped_data = nullptr;

            if (vmaMapMemory(
                graphics::internal::context.allocator,
                vertex_buffer_allocation,
                &mapped_data) != VK_SUCCESS) {
                std::cerr << "Failed to map vertex buffer memory\n";
                return false;
            }

            std::memcpy(mapped_data, vertices.data(), buffer_size);

            vmaFlushAllocation(
                graphics::internal::context.allocator,
                vertex_buffer_allocation,
                0,
                buffer_size);

            vmaUnmapMemory(
                graphics::internal::context.allocator,
                vertex_buffer_allocation);

            return true;
        }

        bool createDescriptorResources() {
            auto& context = graphics::internal::context;

            // Описываем один uniform buffer в descriptor set.
            const VkDescriptorSetLayoutBinding ubo_binding = {
                .binding = 0,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT
            };

            const VkDescriptorSetLayoutCreateInfo layout_info = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .bindingCount = 1,
                .pBindings = &ubo_binding
            };

            if (vkCreateDescriptorSetLayout(
                context.device,
                &layout_info,
                nullptr,
                &descriptor_set_layout) != VK_SUCCESS) {
                std::cerr << "Failed to create descriptor set layout\n";
                return false;
            }

            const VkDescriptorPoolSize pool_size = {
                .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = 2
            };

            const VkDescriptorPoolCreateInfo pool_info = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .maxSets = 2,
                .poolSizeCount = 1,
                .pPoolSizes = &pool_size
            };

            if (vkCreateDescriptorPool(
                context.device,
                &pool_info,
                nullptr,
                &descriptor_pool) != VK_SUCCESS) {
                std::cerr << "Failed to create descriptor pool\n";
                return false;
            }

            // Создаём два отдельных uniform buffer:
            // один для первого объекта, другой для второго.
            for (size_t i = 0; i < uniform_buffers.size(); ++i) {
                const VkBufferCreateInfo buffer_create_info = {
                    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                    .size = sizeof(ObjectData),
                    .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                    .sharingMode = VK_SHARING_MODE_EXCLUSIVE
                };

                const VmaAllocationCreateInfo allocation_create_info = {
                    .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                    .usage = VMA_MEMORY_USAGE_AUTO
                };

                if (vmaCreateBuffer(
                    context.allocator,
                    &buffer_create_info,
                    &allocation_create_info,
                    &uniform_buffers[i],
                    &uniform_buffer_allocations[i],
                    nullptr) != VK_SUCCESS) {
                    std::cerr << "Failed to create uniform buffer\n";
                    return false;
                }
            }

            // Выделяем два descriptor set одного и того же layout.
            std::array<VkDescriptorSetLayout, 2> layouts = {
                descriptor_set_layout,
                descriptor_set_layout
            };

            const VkDescriptorSetAllocateInfo allocate_info = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = descriptor_pool,
                .descriptorSetCount = 2,
                .pSetLayouts = layouts.data()
            };

            if (vkAllocateDescriptorSets(
                context.device,
                &allocate_info,
                descriptor_sets.data()) != VK_SUCCESS) {
                std::cerr << "Failed to allocate descriptor sets\n";
                return false;
            }

            // Каждый descriptor set связываем со своим uniform buffer.
            for (size_t i = 0; i < descriptor_sets.size(); ++i) {
                const VkDescriptorBufferInfo buffer_info = {
                    .buffer = uniform_buffers[i],
                    .offset = 0,
                    .range = sizeof(ObjectData)
                };

                const VkWriteDescriptorSet descriptor_write = {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = descriptor_sets[i],
                    .dstBinding = 0,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .pBufferInfo = &buffer_info
                };

                vkUpdateDescriptorSets(
                    context.device,
                    1,
                    &descriptor_write,
                    0,
                    nullptr);
            }

            return true;
        }

        bool createPipeline() {
            const auto shader_directory = std::filesystem::path("../..") / "shaders";

            const auto vertex_shader_code = readFile((shader_directory / "cone.vert.spv").string());

            const auto fragment_shader_code = readFile((shader_directory / "cone.frag.spv").string());

            const VkShaderModule vertex_shader = createShaderModule(vertex_shader_code);

            const VkShaderModule fragment_shader = createShaderModule(fragment_shader_code);

            if (vertex_shader == VK_NULL_HANDLE ||
                fragment_shader == VK_NULL_HANDLE) {
                std::cerr << "Failed to create shader modules\n";
                return false;
            }

            const VkPipelineShaderStageCreateInfo shader_stages[] = {
                {
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage = VK_SHADER_STAGE_VERTEX_BIT,
                    .module = vertex_shader,
                    .pName = "main"
                },
                {
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                    .module = fragment_shader,
                    .pName = "main"
                }
            };

            const VkVertexInputBindingDescription binding = {
                .binding = 0,
                .stride = sizeof(Vertex),
                .inputRate = VK_VERTEX_INPUT_RATE_VERTEX
            };

            const std::array<VkVertexInputAttributeDescription, 2> attributes = {
                VkVertexInputAttributeDescription{
                    .location = 0,
                    .binding = 0,
                    .format = VK_FORMAT_R32G32B32_SFLOAT,
                    .offset = offsetof(Vertex, position)
                },
                VkVertexInputAttributeDescription{
                    .location = 1,
                    .binding = 0,
                    .format = VK_FORMAT_R32G32B32_SFLOAT,
                    .offset = offsetof(Vertex, color)
                }
            };

            const VkPipelineVertexInputStateCreateInfo vertex_input = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                .vertexBindingDescriptionCount = 1,
                .pVertexBindingDescriptions = &binding,
                .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributes.size()),
                .pVertexAttributeDescriptions = attributes.data()
            };

            const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                .primitiveRestartEnable = VK_FALSE
            };

            const VkPipelineViewportStateCreateInfo viewport_state = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                .viewportCount = 1,
                .scissorCount = 1
            };

            const VkPipelineRasterizationStateCreateInfo rasterization = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                .depthClampEnable = VK_FALSE,
                .rasterizerDiscardEnable = VK_FALSE,
                .polygonMode = VK_POLYGON_MODE_FILL,
                .cullMode = VK_CULL_MODE_NONE,
                .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
                .depthBiasEnable = VK_FALSE,
                .lineWidth = 1.0f
            };

            const VkPipelineMultisampleStateCreateInfo multisampling = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT
            };

            const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                .depthTestEnable = VK_TRUE,
                .depthWriteEnable = VK_TRUE,
                .depthCompareOp = VK_COMPARE_OP_LESS,
                .depthBoundsTestEnable = VK_FALSE,
                .stencilTestEnable = VK_FALSE
            };

            const VkPipelineColorBlendAttachmentState color_blend_attachment = {
                .blendEnable = VK_FALSE,
                .colorWriteMask =
                    VK_COLOR_COMPONENT_R_BIT |
                    VK_COLOR_COMPONENT_G_BIT |
                    VK_COLOR_COMPONENT_B_BIT |
                    VK_COLOR_COMPONENT_A_BIT
            };

            const VkPipelineColorBlendStateCreateInfo color_blend = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                .logicOpEnable = VK_FALSE,
                .attachmentCount = 1,
                .pAttachments = &color_blend_attachment
            };

            const VkDynamicState dynamic_states[] = {
                VK_DYNAMIC_STATE_VIEWPORT,
                VK_DYNAMIC_STATE_SCISSOR
            };

            const VkPipelineDynamicStateCreateInfo dynamic_state = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                .dynamicStateCount = 2,
                .pDynamicStates = dynamic_states
            };

            const VkPipelineLayoutCreateInfo pipeline_layout_info = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                .setLayoutCount = 1,
                .pSetLayouts = &descriptor_set_layout
            };

            if (vkCreatePipelineLayout(
                graphics::internal::context.device,
                &pipeline_layout_info,
                nullptr,
                &pipeline_layout) != VK_SUCCESS) {
                std::cerr << "Failed to create pipeline layout\n";
                return false;
            }

            const VkGraphicsPipelineCreateInfo pipeline_info = {
                .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                .stageCount = 2,
                .pStages = shader_stages,
                .pVertexInputState = &vertex_input,
                .pInputAssemblyState = &input_assembly,
                .pViewportState = &viewport_state,
                .pRasterizationState = &rasterization,
                .pMultisampleState = &multisampling,
                .pDepthStencilState = &depth_stencil,
                .pColorBlendState = &color_blend,
                .pDynamicState = &dynamic_state,
                .layout = pipeline_layout,
                .renderPass = graphics::internal::context.render_pass,
                .subpass = 0
            };

            const VkResult result = vkCreateGraphicsPipelines(
                graphics::internal::context.device,
                VK_NULL_HANDLE,
                1,
                &pipeline_info,
                nullptr,
                &graphics_pipeline);

            vkDestroyShaderModule(
                graphics::internal::context.device,
                vertex_shader,
                nullptr);

            vkDestroyShaderModule(
                graphics::internal::context.device,
                fragment_shader,
                nullptr);

            if (result != VK_SUCCESS) {
                std::cerr << "Failed to create graphics pipeline\n";
                return false;
            }

            return true;
        }

        glm::mat4 getMvp(
            const glm::vec3& position,
            const glm::vec3& rotation,
            const glm::vec3& scale
        ) {
            glm::mat4 model = glm::mat4(1.0f);

            model = glm::translate(model, position);

            model = glm::rotate(
                model,
                rotation.x,
                glm::vec3(1.0f, 0.0f, 0.0f)
            );

            model = glm::rotate(
                model,
                rotation.y,
                glm::vec3(0.0f, 1.0f, 0.0f)
            );

            model = glm::rotate(
                model,
                rotation.z,
                glm::vec3(0.0f, 0.0f, 1.0f)
            );

            model = glm::scale(
                model,
                scale
            );

            const glm::mat4 view = glm::lookAt(
                glm::vec3(0.0f, 0.0f, 3.5f),
                glm::vec3(0.0f, 0.0f, 0.0f),
                glm::vec3(0.0f, 1.0f, 0.0f)
            );

            glm::mat4 projection;

            if (use_perspective) {
                projection = glm::perspective(
                    glm::radians(45.0f),
                    1.0f,
                    0.1f,
                    100.0f
                );
            }
            else {
                projection = glm::ortho(
                    -2.0f,
                    2.0f,
                    -2.0f,
                    2.0f,
                    0.1f,
                    100.0f
                );
            }

            projection[1][1] *= -1.0f;

            return projection * view * model;
        }

    } // namespace

    bool initialize() {

        createCone();

        if (!createVertexBuffer()) {
            return false;
        }

        if (!createDescriptorResources()) {
            return false;
        }

        if (!createPipeline()) {
            return false;
        }

        return true;
    }

    void shutdown() {
        auto& context = graphics::internal::context;

        vkQueueWaitIdle(context.graphics_queue);

        if (graphics_pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(
                context.device,
                graphics_pipeline,
                nullptr);
        }

        if (pipeline_layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(
                context.device,
                pipeline_layout,
                nullptr);
        }

        if (descriptor_pool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(
                context.device,
                descriptor_pool,
                nullptr);
        }

        if (descriptor_set_layout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(
                context.device,
                descriptor_set_layout,
                nullptr);
        }

        if (vertex_buffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(
                context.allocator,
                vertex_buffer,
                vertex_buffer_allocation);
        }

        for (size_t i = 0; i < uniform_buffers.size(); ++i) {
            if (uniform_buffers[i] != VK_NULL_HANDLE) {
                vmaDestroyBuffer(
                    context.allocator,
                    uniform_buffers[i],
                    uniform_buffer_allocations[i]);
            }
        }
    }

    void update(double time) {
        ImGui::Begin("Cone controls");

        ImGui::Checkbox("Perspective", &use_perspective);

        ImGui::Separator();

        ImGui::DragFloat3(
            "Position",
            &object_position.x,
            0.01f
        );

        ImGui::DragFloat3(
            "Rotation",
            &object_rotation.x,
            0.01f
        );

        ImGui::DragFloat3(
            "Scale",
            &object_scale.x,
            0.01f,
            0.01f,
            5.0f
        );

        ImGui::Separator();

        ImGui::Checkbox("Play animation", &animation_enabled);

        ImGui::DragFloat(
            "Animation speed",
            &animation_speed,
            0.01f,
            0.0f,
            5.0f
        );

        ImGui::DragFloat(
            "Trajectory radius",
            &trajectory_radius,
            0.01f,
            0.1f,
            5.0f
        );

        ImGui::Separator();

        ImGui::ColorEdit3(
            "Object color",
            &object_color.x
        );

        if (animation_enabled) {
            const float animation_time = static_cast<float>(time * animation_speed);

            object_position.x = trajectory_radius * std::cos(animation_time);
            object_position.z = trajectory_radius * std::sin(animation_time);
            object_position.y = 0.5f * std::sin(2.0f * animation_time);
            object_rotation.y = animation_time;
            const float second_animation_time = animation_time + 3.14159265358979323846f;
            second_object_position.x = trajectory_radius * std::cos(second_animation_time);
            second_object_position.z = trajectory_radius * std::sin(second_animation_time);
            second_object_position.y = 0.5f * std::sin(2.0f * second_animation_time);
            second_object_rotation.y = second_animation_time;
        }

        ImGui::End();
    }

    void updateUniformBuffer(size_t index, const ObjectData& data) {
        auto& context = graphics::internal::context;

        void* mapped_data = nullptr;

        if (vmaMapMemory(
            context.allocator,
            uniform_buffer_allocations[index],
            &mapped_data) != VK_SUCCESS) {
            return;
        }

        std::memcpy(mapped_data, &data, sizeof(ObjectData));

        vmaFlushAllocation(
            context.allocator,
            uniform_buffer_allocations[index],
            0,
            sizeof(ObjectData));

        vmaUnmapMemory(
            context.allocator,
            uniform_buffer_allocations[index]);
    }

    void render(const graphics::internal::FrameData& fd) {
        auto& context = graphics::internal::context;

        const VkCommandBuffer command_buffer = fd.command_buffer;

        vkResetCommandBuffer(command_buffer, 0);

        const VkCommandBufferBeginInfo begin_info = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        };

        vkBeginCommandBuffer(command_buffer, &begin_info);

        const VkClearValue clear_values[] = {
            {.color = {{0.03f, 0.04f, 0.06f, 1.0f}}},
            {.depthStencil = {1.0f, 0}}
        };

        const VkRenderPassBeginInfo render_pass_begin = {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = context.render_pass,
            .framebuffer = fd.framebuffer,
            .renderArea = {
                .offset = {0, 0},
                .extent = context.swapchain_extent
            },
            .clearValueCount = 2,
            .pClearValues = clear_values
        };

        vkCmdBeginRenderPass(
            command_buffer,
            &render_pass_begin,
            VK_SUBPASS_CONTENTS_INLINE);

        const VkViewport viewport = {
            .x = 0.0f,
            .y = 0.0f,
            .width = static_cast<float>(context.swapchain_extent.width),
            .height = static_cast<float>(context.swapchain_extent.height),
            .minDepth = 0.0f,
            .maxDepth = 1.0f
        };

        const VkRect2D scissor = {
            .offset = {0, 0},
            .extent = context.swapchain_extent
        };

        vkCmdSetViewport(command_buffer, 0, 1, &viewport);

        vkCmdSetScissor(command_buffer, 0, 1, &scissor);

        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline);

        const VkDeviceSize offset = 0;

        vkCmdBindVertexBuffers(command_buffer, 0, 1, &vertex_buffer, &offset);

        // Первый конус
        ObjectData first_object{};
        first_object.mvp = getMvp(object_position, object_rotation, object_scale);
        first_object.color = glm::vec4(object_color, 1.0f);

        updateUniformBuffer(0, first_object);

        vkCmdBindDescriptorSets(
            command_buffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipeline_layout,
            0,
            1,
            &descriptor_sets[0],
            0,
            nullptr
        );

        vkCmdDraw(command_buffer, static_cast<uint32_t>(vertices.size()), 1, 0,0);

		// Второй конус
        ObjectData second_object{};

        second_object.mvp = getMvp(second_object_position, second_object_rotation, object_scale);

        second_object.color = glm::vec4(object_color, 1.0f);

        updateUniformBuffer(1, second_object);

        vkCmdBindDescriptorSets(
            command_buffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipeline_layout,
            0,
            1,
            &descriptor_sets[1],
            0,
            nullptr
        );

        vkCmdDraw(command_buffer, static_cast<uint32_t>(vertices.size()), 1, 0, 0);

        vkCmdEndRenderPass(command_buffer);

        vkEndCommandBuffer(command_buffer);
    }

} // namespace application