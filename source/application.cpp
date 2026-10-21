#include "application.hpp"

#include <imgui.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace application {
    namespace {
        constexpr double pi = 3.14159265358979323846;
        constexpr uint32_t cone_index_count = 192;

        struct Vertex {
            glm::vec3 position;
            glm::vec3 color;
            glm::vec3 normal;
        };

        struct alignas(16) SceneData {
            glm::mat4 view_projection;
            glm::vec4 camera_position;
            glm::vec4 camera_direction;
            glm::vec4 ambient; // Цвет RGB и интенсивность фонового освещения
            glm::vec4 light_direction;
            glm::vec4 light_color; // Цвет RGB и интенсивность (ноль при отключённом источнике)
            glm::ivec4 counts; // Число точечных источников, число прожекторов, флаг перспективы, резерв
        };
        struct alignas(16) ObjectData {
            glm::mat4 model;
            glm::mat4 normal_matrix;
            glm::vec4 albedo;
            glm::vec4 specular;
            glm::vec4 parameters; // Степень блеска, флаг использования цветов вершин, два резервных поля
        };
        struct alignas(16) PointData {
            glm::vec4 position;
            glm::vec4 color;
        };
        struct alignas(16) SpotData {
            glm::vec4 position;
            glm::vec4 color;
            glm::vec4 direction;
            glm::vec4 angles; // Косинусы внутреннего и внешнего полууглов, два резервных поля
        };
        
        static_assert(sizeof(SceneData) == 160 && offsetof(SceneData, counts) == 144);
        static_assert(offsetof(SceneData, camera_position) == 64 && offsetof(SceneData, camera_direction) == 80);
        
        static_assert(offsetof(SceneData, ambient) == 96 && offsetof(SceneData, light_direction) == 112
            && offsetof(SceneData, light_color) == 128);
        
        static_assert(sizeof(ObjectData) == 176 && offsetof(ObjectData, normal_matrix) == 64);
        
        static_assert(offsetof(ObjectData, albedo) == 128 && offsetof(ObjectData, specular) == 144
            && offsetof(ObjectData, parameters) == 160);
        
        static_assert(sizeof(PointData) == 32 && offsetof(PointData, color) == 16);
        
        static_assert(sizeof(SpotData) == 64 && offsetof(SpotData, color) == 16
            && offsetof(SpotData, direction) == 32 && offsetof(SpotData, angles) == 48);

        struct Material {
            glm::vec3 albedo{0.2f, 0.5f, 1.0f};
            glm::vec3 specular{0.7f};
            float shininess = 64.0f;
            bool vertex_colors = false;
        };
        
        struct PointLight {
            glm::vec3 position{2.0f, 2.0f, 2.0f};
            glm::vec3 color{1.0f, 0.55f, 0.25f};
            float intensity = 12.0f;
            bool enabled = true;
        };
        
        struct SpotLight {
            glm::vec3 position{-2.0f, 4.0f, 1.0f};
            glm::vec3 color{0.4f, 0.65f, 1.0f};
            float intensity = 25.0f;
            bool enabled = true;
            glm::vec3 direction{0.25f, -1.0f, -0.15f};
            float inner_angle = 18.0f;
            float outer_angle = 30.0f;
        };
        
        struct Buffer {
            VkBuffer handle = VK_NULL_HANDLE;
            VmaAllocation allocation = VK_NULL_HANDLE;
            VkDeviceSize size = 0;
        };
        
        Buffer vertex_buffer, index_buffer, scene_buffer, point_buffer, spot_buffer;
        std::array<Buffer, 3> model_buffers{};
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        VkPipeline graphics_pipeline = VK_NULL_HANDLE;
        VkDescriptorSetLayout scene_layout = VK_NULL_HANDLE;
        VkDescriptorSetLayout model_layout = VK_NULL_HANDLE;
        VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
        VkDescriptorSet scene_set = VK_NULL_HANDLE;
        std::array<VkDescriptorSet, 3> model_sets{};
        VkDeviceSize max_storage_range = 0;
        const char* resource_error = nullptr;

        glm::vec3 object_position{0.0f}, object_rotation{0.0f}, object_scale{1.0f};
        glm::vec3 object_color{1.0f}; // Общий оттенок, который умножается на альбедо каждого конуса
        glm::vec3 first_object_position{0.0f}, first_object_rotation{0.0f};
        glm::vec3 second_object_position{0.0f}, second_object_rotation{0.0f};
        std::array<Material, 3> materials{};
        bool show_floor = true;

        glm::vec3 camera_position{0.0f, 2.0f, 8.0f};
        float camera_yaw = -90.0f, camera_pitch = -12.0f;
        float camera_fov = 45.0f, ortho_size = 3.5f;
        bool use_perspective = true;

        glm::vec3 ambient_color{1.0f};
        float ambient_intensity = 0.08f;
        bool directional_enabled = true;
        glm::vec3 directional_direction{-0.5f, -1.0f, -0.5f};
        glm::vec3 directional_color{1.0f};
        float directional_intensity = 0.6f;
        std::vector<PointLight> point_lights{PointLight{}};
        std::vector<SpotLight> spot_lights{SpotLight{}};

        bool animation_enabled = true;
        float animation_speed = 1.0f, trajectory_radius = 1.5f;
        double animation_phase = 0.0;
        bool animate_light = false, light_animation_playing = true;
        float light_animation_speed = 1.0f, light_animation_radius = 1.5f;
        double light_phase = 0.0;
        double previous_update_time = 0.0;
        bool has_previous_update_time = false;
        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;

        glm::vec3 safeDirection(const glm::vec3& value) {
            const float length_squared = glm::dot(value, value);
            return length_squared > 1e-8f ? value / std::sqrt(length_squared) : glm::vec3(0, -1, 0);
        }

        void destroyBuffer(Buffer& buffer) {
            if (buffer.handle) {
                vmaDestroyBuffer(graphics::internal::context.allocator, buffer.handle, buffer.allocation);
            }
            buffer = {};
        }

        bool createBuffer(Buffer& buffer, VkDeviceSize size, VkBufferUsageFlags usage) {
            const VkBufferCreateInfo info = {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .size = size,
                .usage = usage,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE
            };
            const VmaAllocationCreateInfo allocation_info = {
                .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                .usage = VMA_MEMORY_USAGE_AUTO
            };
            if (vmaCreateBuffer(graphics::internal::context.allocator, &info, &allocation_info,
                &buffer.handle, &buffer.allocation, nullptr) != VK_SUCCESS) {
                resource_error = "Failed to allocate GPU buffer; scene rendering skipped.";
                std::cerr << resource_error << '\n';
                return false;
            }
            buffer.size = size;
            return true;
        }

        bool upload(const Buffer& buffer, const void* data, VkDeviceSize size) {
            if (size == 0) return true;
            auto allocator = graphics::internal::context.allocator;
            void* mapped = nullptr;
            if (size > buffer.size || vmaMapMemory(allocator, buffer.allocation, &mapped) != VK_SUCCESS) {
                resource_error = "Failed to map GPU buffer; scene rendering skipped.";
                return false;
            }
            std::memcpy(mapped, data, static_cast<size_t>(size));
            const VkResult result = vmaFlushAllocation(allocator, buffer.allocation, 0, size);
            vmaUnmapMemory(allocator, buffer.allocation);
            if (result != VK_SUCCESS) resource_error = "Failed to flush GPU buffer; scene rendering skipped.";
            return result == VK_SUCCESS;
        }

        void bindBuffer(VkDescriptorSet set, uint32_t binding, VkDescriptorType type, const Buffer& buffer) {
            const VkDescriptorBufferInfo info = {buffer.handle, 0, buffer.size};
            const VkWriteDescriptorSet write = {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = set,
                .dstBinding = binding,
                .descriptorCount = 1,
                .descriptorType = type,
                .pBufferInfo = &info
            };
            vkUpdateDescriptorSets(graphics::internal::context.device, 1, &write, 0, nullptr);
        }

        bool ensureStorage(Buffer& buffer, VkDeviceSize required, uint32_t binding) {
            if (required > max_storage_range) {
                resource_error = "Light array exceeds the device storage buffer limit.";
                return false;
            }
            if (buffer.size >= required) return true;
            // Сначала создаём новый буфер: при ошибке выделения памяти старый должен сохраниться.
            Buffer replacement;
            if (!createBuffer(replacement, required, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) return false;
            bindBuffer(scene_set, binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, replacement);
            destroyBuffer(buffer);
            buffer = replacement;
            return true;
        }

        void createCone() {
            constexpr uint32_t segments = 32;
            constexpr float radius = 1.0f, half_height = 1.0f;
            vertices.clear();
            indices.clear();
            vertices.reserve(101); // 97 вершин конуса и 4 вершины пола
            indices.reserve(cone_index_count + 6);
            const auto color = [](glm::vec3 p) { return 0.5f + 0.5f * p; };
            const auto sideNormal = [](float angle) {
                return glm::normalize(glm::vec3(std::cos(angle), radius / (2 * half_height), std::sin(angle)));
            };

            for (uint32_t i = 0; i < segments; ++i) {
                const float angle = static_cast<float>(2 * pi * i / segments);
                const float middle = static_cast<float>(2 * pi * (i + 0.5) / segments);
                const glm::vec3 rim(radius * std::cos(angle), -half_height, radius * std::sin(angle));
                const glm::vec3 apex(0, half_height, 0);
                // Одинаковые позиции, разные нормали: край основания остаётся чётким.
                vertices.push_back({rim, color(rim), sideNormal(angle)});
                vertices.push_back({rim, color(rim), {0, -1, 0}});
                vertices.push_back({apex, color(apex), sideNormal(middle)});
            }

            vertices.push_back({{0, -half_height, 0}, {0.5f, 0, 0.5f}, {0, -1, 0}});
            for (uint32_t i = 0; i < segments; ++i) {
                const uint32_t next = (i + 1) % segments;
                // Порядок обхода вершин задаёт внешнюю сторону: наружу для боковой поверхности и вниз для основания.
                indices.insert(indices.end(), {3*i+2, 3*next, 3*i, 96, 3*i+1, 3*next+1});
            }

            for (const glm::vec3 p : std::array<glm::vec3, 4>{
                glm::vec3(-10, -2, -10), {10, -2, -10}, {10, -2, 10}, {-10, -2, 10}}) {
                vertices.push_back({p, {1, 1, 1}, {0, 1, 0}});
            }

            indices.insert(indices.end(), {97, 99, 98, 97, 100, 99});
        }

        bool createDescriptorResources() {
            auto device = graphics::internal::context.device;

            const VkDescriptorSetLayoutBinding scene_bindings[] = {
                {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}
            };

            VkDescriptorSetLayoutCreateInfo layout_info = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .bindingCount = 3,
                .pBindings = scene_bindings
            };

            if (vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &scene_layout) != VK_SUCCESS) return false;
            layout_info.bindingCount = 1;
            
            if (vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &model_layout) != VK_SUCCESS) return false;
            
            const VkDescriptorPoolSize sizes[] = {
                {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2}
            };
            
            const VkDescriptorPoolCreateInfo pool_info = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .maxSets = 4,
                .poolSizeCount = 2,
                .pPoolSizes = sizes
            };
            
            if (vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) return false;
            const VkDescriptorSetLayout layouts[] = {scene_layout, model_layout, model_layout, model_layout};
            const VkDescriptorSetAllocateInfo allocation_info = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = descriptor_pool,
                .descriptorSetCount = 4,
                .pSetLayouts = layouts
            };
            std::array<VkDescriptorSet, 4> sets{};
            
            if (vkAllocateDescriptorSets(device, &allocation_info, sets.data()) != VK_SUCCESS) return false;
            scene_set = sets[0];
            
            if (!createBuffer(scene_buffer, sizeof(SceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)) return false;
            bindBuffer(scene_set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, scene_buffer);
            
            for (size_t i = 0; i < model_sets.size(); ++i) {
                model_sets[i] = sets[i + 1];
                if (!createBuffer(model_buffers[i], sizeof(ObjectData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)) return false;
                bindBuffer(model_sets[i], 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, model_buffers[i]);
            }
            return ensureStorage(point_buffer, sizeof(PointData), 1)
                && ensureStorage(spot_buffer, sizeof(SpotData), 2);
        }

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

        bool createPipeline() {
            // Поддерживаем запуск из корня репозитория и из папки Debug, созданной Visual Studio.
            auto shader_directory = std::filesystem::path("shaders");
            if (!std::filesystem::exists(shader_directory / "cone.vert.spv")) {
                shader_directory = std::filesystem::path("../..") / "shaders";
            }
#ifdef _WIN32
            if (!std::filesystem::exists(shader_directory / "cone.vert.spv")) {
                // При запуске вне Visual Studio рабочая папка может не совпадать с папками проекта.
                std::array<wchar_t, 32768> executable_path{};
                const DWORD length = GetModuleFileNameW(nullptr, executable_path.data(),
                    static_cast<DWORD>(executable_path.size()));
                if (length > 0 && length < executable_path.size()) {
                    shader_directory = std::filesystem::path(executable_path.data()).parent_path()
                        / "../.." / "shaders";
                }
            }
#endif

            const auto vertex_shader_code = readFile((shader_directory / "cone.vert.spv").string());

            const auto fragment_shader_code = readFile((shader_directory / "cone.frag.spv").string());

            const VkShaderModule vertex_shader = createShaderModule(vertex_shader_code);

            const VkShaderModule fragment_shader = createShaderModule(fragment_shader_code);

            const auto destroy_shader_modules = [&]() {
                vkDestroyShaderModule(graphics::internal::context.device, vertex_shader, nullptr);
                vkDestroyShaderModule(graphics::internal::context.device, fragment_shader, nullptr);
            };

            if (vertex_shader == VK_NULL_HANDLE ||
                fragment_shader == VK_NULL_HANDLE) {
                std::cerr << "Failed to create shader modules\n";
                destroy_shader_modules();
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

            const std::array<VkVertexInputAttributeDescription, 3> attributes = {
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
                },
                VkVertexInputAttributeDescription{
                    .location = 2,
                    .binding = 0,
                    .format = VK_FORMAT_R32G32B32_SFLOAT,
                    .offset = offsetof(Vertex, normal)
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

            const VkDescriptorSetLayout layouts[] = { scene_layout, model_layout };
            const VkPipelineLayoutCreateInfo pipeline_layout_info = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                .setLayoutCount = 2,
                .pSetLayouts = layouts
            };

            if (vkCreatePipelineLayout(
                graphics::internal::context.device,
                &pipeline_layout_info,
                nullptr,
                &pipeline_layout) != VK_SUCCESS) {
                std::cerr << "Failed to create pipeline layout\n";
                destroy_shader_modules();
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

            destroy_shader_modules();

            if (result != VK_SUCCESS) {
                std::cerr << "Failed to create graphics pipeline\n";
                return false;
            }

            return true;
        }


        glm::mat4 getModel(const glm::vec3& position, const glm::vec3& rotation, const glm::vec3& scale) {
            glm::mat4 model = glm::translate(glm::mat4(1), position);
            model = glm::rotate(model, rotation.x, glm::vec3(1, 0, 0));
            model = glm::rotate(model, rotation.y, glm::vec3(0, 1, 0));
            model = glm::rotate(model, rotation.z, glm::vec3(0, 0, 1));
            return glm::scale(model, scale);
        }

        bool updateGpuData() {
            resource_error = nullptr;
            const VkDeviceSize point_size = sizeof(PointData) * std::max(size_t(1), point_lights.size());
            const VkDeviceSize spot_size = sizeof(SpotData) * std::max(size_t(1), spot_lights.size());
            
            if (!ensureStorage(point_buffer, point_size, 1) || !ensureStorage(spot_buffer, spot_size, 2)) return false;
            std::vector<PointData> points;
            
            for (size_t i = 0; i < point_lights.size(); ++i) {
                const auto& light = point_lights[i];
                glm::vec3 position = light.position;
                
                if (i == 0 && animate_light) {
                    position += light_animation_radius * glm::vec3(
                        static_cast<float>(std::cos(light_phase)), 0, static_cast<float>(std::sin(light_phase)));
                }
                
                points.push_back({glm::vec4(position, 1), glm::vec4(light.color, light.enabled ? light.intensity : 0)});
            }
            std::vector<SpotData> spots;
            
            for (const auto& light : spot_lights) {
                spots.push_back({glm::vec4(light.position, 1), glm::vec4(light.color, light.enabled ? light.intensity : 0),
                    glm::vec4(safeDirection(light.direction), 0),
                    glm::vec4(std::cos(glm::radians(light.inner_angle)), std::cos(glm::radians(light.outer_angle)), 0, 0)});
            }
            
            if (!upload(point_buffer, points.data(), sizeof(PointData) * points.size())
                || !upload(spot_buffer, spots.data(), sizeof(SpotData) * spots.size())) return false;

            const float yaw = glm::radians(camera_yaw), pitch = glm::radians(camera_pitch);
            const glm::vec3 forward(std::cos(pitch) * std::cos(yaw), std::sin(pitch), std::cos(pitch) * std::sin(yaw));
            const glm::mat4 view = glm::lookAt(camera_position, camera_position + forward, glm::vec3(0, 1, 0));
            const auto extent = graphics::internal::context.swapchain_extent;
            const float aspect = extent.width && extent.height ? static_cast<float>(extent.width) / extent.height : 1.0f;
            
            glm::mat4 projection = use_perspective
                ? glm::perspective(glm::radians(camera_fov), aspect, 0.1f, 100.0f)
                : glm::ortho(-ortho_size * aspect, ortho_size * aspect, -ortho_size, ortho_size, 0.1f, 100.0f);
            
            projection[1][1] *= -1;
            
            const SceneData scene{
                projection * view, glm::vec4(camera_position, 1), glm::vec4(forward, 0),
                glm::vec4(ambient_color, ambient_intensity), glm::vec4(safeDirection(directional_direction), 0),
                glm::vec4(directional_color, directional_enabled ? directional_intensity : 0),
                glm::ivec4(static_cast<int>(points.size()), static_cast<int>(spots.size()), use_perspective ? 1 : 0, 0)
            };
            
            if (!upload(scene_buffer, &scene, sizeof(scene))) return false;
            const std::array<glm::mat4, 3> models = {
                getModel(first_object_position, first_object_rotation, object_scale),
                getModel(second_object_position, second_object_rotation, object_scale), glm::mat4(1)
            };
            
            for (size_t i = 0; i < models.size(); ++i) {
                const auto& material = materials[i];
                const ObjectData data{models[i], glm::transpose(glm::inverse(models[i])),
                    glm::vec4(material.albedo * (i == 2 ? glm::vec3(1) : object_color), 1),
                    glm::vec4(material.specular, 1), glm::vec4(material.shininess, material.vertex_colors ? 1 : 0, 0, 0)};
                if (!upload(model_buffers[i], &data, sizeof(data))) return false;
            }
            return true;
        }

        void materialControls(const char* title, Material& material) {
            if (ImGui::TreeNode(title)) {
                ImGui::ColorEdit3("Albedo", &material.albedo.x);
                ImGui::ColorEdit3("Specular color", &material.specular.x);
                ImGui::SliderFloat("Shininess", &material.shininess, 1, 256);
                ImGui::Checkbox("Procedural vertex colors", &material.vertex_colors);
                ImGui::TreePop();
            }
        }

        void lightControls() {
            ImGui::ColorEdit3("Ambient color", &ambient_color.x);
            ImGui::SliderFloat("Ambient intensity", &ambient_intensity, 0, 1);
            if (ImGui::TreeNode("Directional light")) {
                ImGui::Checkbox("Enabled", &directional_enabled);
                ImGui::DragFloat3("Direction (rays)", &directional_direction.x, 0.01f);
                ImGui::ColorEdit3("Color", &directional_color.x);
                ImGui::SliderFloat("Intensity", &directional_intensity, 0, 5);
                ImGui::TextWrapped("Directions are normalized. Zero direction uses (0, -1, 0).");
                ImGui::TreePop();
            }
            
            if (ImGui::TreeNode("Point lights")) {
                ImGui::BeginDisabled(point_lights.size() >= max_storage_range / sizeof(PointData));
                
                if (ImGui::Button("Add point light")) point_lights.emplace_back();
                ImGui::EndDisabled();
                
                for (size_t i = 0; i < point_lights.size();) {
                    ImGui::PushID(static_cast<int>(i));
                    const bool open = ImGui::TreeNode("Point", "Point %d", static_cast<int>(i + 1));
                    bool remove = false;
                    if (open) {
                        auto& light = point_lights[i];
                        ImGui::Checkbox("Enabled", &light.enabled);
                        ImGui::DragFloat3("Position", &light.position.x, 0.05f);
                        ImGui::ColorEdit3("Color", &light.color.x);
                        ImGui::SliderFloat("Intensity", &light.intensity, 0, 100);
                        remove = ImGui::Button("Delete");
                        ImGui::TreePop();
                    }
                    
                    ImGui::PopID();
                    if (remove) point_lights.erase(point_lights.begin() + i);
                    else ++i;
                }
                ImGui::BeginDisabled(point_lights.empty());
                ImGui::Checkbox("Animate first point light", &animate_light);
                ImGui::Checkbox("Play light animation", &light_animation_playing);
                ImGui::SliderFloat("Light animation speed", &light_animation_speed, 0, 5);
                ImGui::SliderFloat("Light orbit radius", &light_animation_radius, 0, 5);
                ImGui::EndDisabled();
                ImGui::TreePop();
            }
            
            if (ImGui::TreeNode("Spot lights")) {
                ImGui::BeginDisabled(spot_lights.size() >= max_storage_range / sizeof(SpotData));
                
                if (ImGui::Button("Add spot light")) spot_lights.emplace_back();
                ImGui::EndDisabled();
                
                for (size_t i = 0; i < spot_lights.size();) {
                    ImGui::PushID(static_cast<int>(i));
                    const bool open = ImGui::TreeNode("Spot", "Spot %d", static_cast<int>(i + 1));
                    bool remove = false;
                    
                    if (open) {
                        auto& light = spot_lights[i];
                        ImGui::Checkbox("Enabled", &light.enabled);
                        ImGui::DragFloat3("Position", &light.position.x, 0.05f);
                        ImGui::DragFloat3("Direction (rays)", &light.direction.x, 0.01f);
                        ImGui::ColorEdit3("Color", &light.color.x);
                        ImGui::SliderFloat("Intensity", &light.intensity, 0, 100);
                        ImGui::SliderFloat("Inner half-angle (deg)", &light.inner_angle, 0, 89);
                        ImGui::SliderFloat("Outer half-angle (deg)", &light.outer_angle, 0, 89);
                        light.inner_angle = std::clamp(light.inner_angle, 0.0f, 89.0f);
                        light.outer_angle = std::clamp(light.outer_angle, light.inner_angle, 89.0f);
                        remove = ImGui::Button("Delete");
                        ImGui::TreePop();
                    }
                    
                    ImGui::PopID();
                    if (remove) spot_lights.erase(spot_lights.begin() + i);
                    else ++i;
                }
                ImGui::TreePop();
            }
        }
    } // namespace

    bool initialize() {
        animation_phase = light_phase = 0;
        has_previous_update_time = false;
        materials[1].albedo = glm::vec3(1.0f, 0.35f, 0.12f);
        materials[2].albedo = glm::vec3(0.45f);
        materials[2].specular = glm::vec3(0.15f);
        materials[2].shininess = 32;
        
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(graphics::internal::context.physical_device, &properties);
        max_storage_range = properties.limits.maxStorageBufferRange;
        createCone();
        
        if (!createBuffer(vertex_buffer, sizeof(Vertex) * vertices.size(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)
            || !upload(vertex_buffer, vertices.data(), vertex_buffer.size)
            || !createBuffer(index_buffer, sizeof(uint32_t) * indices.size(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT)
            || !upload(index_buffer, indices.data(), index_buffer.size)
            || !createDescriptorResources() || !createPipeline()) {
            std::cerr << "Failed to initialize application resources\n";
            shutdown();
            return false;
        }
        return true;
    }

    void shutdown() {
        auto& context = graphics::internal::context;
        vkQueueWaitIdle(context.graphics_queue);
        
        if (graphics_pipeline) vkDestroyPipeline(context.device, graphics_pipeline, nullptr);
        if (pipeline_layout) vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
        if (descriptor_pool) vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
        if (scene_layout) vkDestroyDescriptorSetLayout(context.device, scene_layout, nullptr);
        if (model_layout) vkDestroyDescriptorSetLayout(context.device, model_layout, nullptr);
        
        graphics_pipeline = VK_NULL_HANDLE;
        pipeline_layout = VK_NULL_HANDLE;
        descriptor_pool = VK_NULL_HANDLE;
        scene_layout = model_layout = VK_NULL_HANDLE;
        scene_set = VK_NULL_HANDLE;
        model_sets.fill(VK_NULL_HANDLE);
        
        for (auto& buffer : model_buffers) destroyBuffer(buffer);
        
        destroyBuffer(scene_buffer);
        destroyBuffer(point_buffer);
        destroyBuffer(spot_buffer);
        destroyBuffer(vertex_buffer);
        destroyBuffer(index_buffer);
    }

    void update(double time) {
        const double delta_time = has_previous_update_time ? std::max(0.0, time - previous_update_time) : 0;
        previous_update_time = time;
        has_previous_update_time = true;
        ImGui::SetNextWindowSize(ImVec2(370, 650), ImGuiCond_FirstUseEver);
        ImGui::Begin("Lab 2: Camera and lighting");
        
        if (resource_error) ImGui::TextWrapped("%s", resource_error);
        
        if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::DragFloat3("Camera position", &camera_position.x, 0.05f);
            ImGui::DragFloat("Yaw (deg)", &camera_yaw, 0.5f);
            ImGui::SliderFloat("Pitch (deg)", &camera_pitch, -89, 89);
            ImGui::Checkbox("Perspective", &use_perspective);
            ImGui::SliderFloat("Field of view (deg)", &camera_fov, 20, 100);
            ImGui::SliderFloat("Ortho half-height", &ortho_size, 0.5f, 20);
            
            if (ImGui::Button("Reset camera")) {
                camera_position = glm::vec3(0, 2, 8);
                camera_yaw = -90; camera_pitch = -12; camera_fov = 45; ortho_size = 3.5f;
                use_perspective = true;
            }
        }
        
        if (ImGui::CollapsingHeader("Models")) {
            ImGui::DragFloat3("Position offset (first)", &object_position.x, 0.01f);
            ImGui::DragFloat3("Rotation offset (rad, first)", &object_rotation.x, 0.01f);
            ImGui::DragFloat3("Scale (both)", &object_scale.x, 0.01f, 0.01f, 5);
            ImGui::ColorEdit3("Shared color tint", &object_color.x);
            ImGui::Checkbox("Play cone animation", &animation_enabled);
            ImGui::SliderFloat("Cone animation speed", &animation_speed, 0, 5);
            ImGui::SliderFloat("Trajectory radius", &trajectory_radius, 0.1f, 5);
            ImGui::Checkbox("Show floor", &show_floor);
        }
        
        if (ImGui::CollapsingHeader("Materials")) {
            materialControls("First cone", materials[0]);
            materialControls("Second cone", materials[1]);
            materialControls("Floor", materials[2]);
        }
        
        if (ImGui::CollapsingHeader("Lighting", ImGuiTreeNodeFlags_DefaultOpen)) lightControls();
        
        ImGui::End();
        
        // Ограничиваем также значения, введённые вручную, в том числе за пределами диапазонов ползунков.
        camera_pitch = std::clamp(camera_pitch, -89.0f, 89.0f);
        camera_fov = std::clamp(camera_fov, 20.0f, 100.0f);
        ortho_size = std::clamp(ortho_size, 0.5f, 20.0f);
        object_scale = glm::clamp(object_scale, glm::vec3(0.01f), glm::vec3(5));
        animation_speed = std::clamp(animation_speed, 0.0f, 5.0f);
        trajectory_radius = std::clamp(trajectory_radius, 0.1f, 5.0f);
        light_animation_speed = std::clamp(light_animation_speed, 0.0f, 5.0f);
        light_animation_radius = std::clamp(light_animation_radius, 0.0f, 5.0f);
        ambient_intensity = std::clamp(ambient_intensity, 0.0f, 1.0f);
        directional_intensity = std::clamp(directional_intensity, 0.0f, 5.0f);
        
        for (auto& light : point_lights) light.intensity = std::clamp(light.intensity, 0.0f, 100.0f);
        for (auto& light : spot_lights) light.intensity = std::clamp(light.intensity, 0.0f, 100.0f);
        for (auto& material : materials) material.shininess = std::clamp(material.shininess, 1.0f, 256.0f);
        
        if (animation_enabled) animation_phase = std::fmod(animation_phase + delta_time * animation_speed, 2 * pi);
        
        if (animate_light && light_animation_playing && !point_lights.empty()) {
            light_phase = std::fmod(light_phase + delta_time * light_animation_speed, 2 * pi);
        }
        
        const auto trajectory = [](double phase) {
            return glm::vec3(static_cast<float>(trajectory_radius * std::cos(phase)),
                static_cast<float>(0.5 * std::sin(2 * phase)), static_cast<float>(trajectory_radius * std::sin(phase)));
        };
        
        first_object_position = object_position + trajectory(animation_phase);
        first_object_rotation = object_rotation + glm::vec3(0, static_cast<float>(animation_phase), 0);
        second_object_position = trajectory(animation_phase + pi);
        second_object_rotation = glm::vec3(0, static_cast<float>(animation_phase + pi), 0);
    }

    void render(const graphics::internal::FrameData& fd) {
        auto& context = graphics::internal::context;

        const bool data_ready = updateGpuData();
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

        vkCmdBindVertexBuffers(command_buffer, 0, 1, &vertex_buffer.handle, &offset);

        vkCmdBindIndexBuffer(command_buffer, index_buffer.handle, 0, VK_INDEX_TYPE_UINT32);

        if (data_ready) {
            vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                pipeline_layout, 0, 1, &scene_set, 0, nullptr);
            
            for (size_t i = 0; i < model_sets.size(); ++i) {
                if (i == 2 && !show_floor) continue;
                
                vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    pipeline_layout, 1, 1, &model_sets[i], 0, nullptr);
                
                vkCmdDrawIndexed(command_buffer, i == 2 ? 6u : cone_index_count,
                    1, i == 2 ? cone_index_count : 0u, 0, 0);
            }
        }

        vkCmdEndRenderPass(command_buffer);

        vkEndCommandBuffer(command_buffer);
    }

} // namespace application
