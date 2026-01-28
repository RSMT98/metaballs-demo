#ifdef WIN32
#include <SDL.h>
#undef main
#else
#include <SDL2/SDL.h>
#endif

#include <GL/glew.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

std::string to_string(std::string_view str) { return std::string(str.begin(), str.end()); }

void sdl2_fail(std::string_view message) { throw std::runtime_error(to_string(message) + SDL_GetError()); }

void glew_fail(std::string_view message, GLenum error) {
    throw std::runtime_error(to_string(message) + reinterpret_cast<const char*>(glewGetErrorString(error)));
}

GLuint create_shader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint status = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status != GL_TRUE) {
        GLint log_length = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_length);
        std::string log(log_length, '\0');
        glGetShaderInfoLog(shader, log_length, nullptr, log.data());
        throw std::runtime_error("Shader compilation failed: " + log);
    }
    return shader;
}

GLuint create_program(const std::vector<GLuint>& shaders) {
    GLuint program = glCreateProgram();
    for (GLuint shader : shaders) glAttachShader(program, shader);
    glLinkProgram(program);

    GLint status = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (status != GL_TRUE) {
        GLint log_length = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &log_length);
        std::string log(log_length, '\0');
        glGetProgramInfoLog(program, log_length, nullptr, log.data());
        throw std::runtime_error("Program link failed: " + log);
    }
    return program;
}

const char vertex_shader_source[] = R"(#version 330 core
uniform ivec3 cell_grid;

flat out ivec3 cell;

void main()
{
    int id = gl_VertexID;
    int x = id % cell_grid.x;
    int y = (id / cell_grid.x) % cell_grid.y;
    int z = id / (cell_grid.x * cell_grid.y);
    cell = ivec3(x, y, z);
    gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
}
)";

const char geometry_shader_source[] = R"(#version 330 core
layout(points) in;
layout(triangle_strip, max_vertices = 36) out;

uniform sampler3D field_tex;
uniform float iso_level;
uniform ivec3 grid_size;
uniform vec3 volume_min;
uniform vec3 volume_max;
uniform mat4 view;
uniform mat4 projection;

flat in ivec3 cell[];

out vec3 world_pos;
out vec3 world_normal;
out vec3 albedo;

vec3 field_gradient(vec3 world_pos)
{
    vec3 tex = clamp((world_pos - volume_min) / (volume_max - volume_min), vec3(0.0), vec3(1.0));
    vec3 texel = 1.0 / vec3(grid_size);

    float fx1 = texture(field_tex, clamp(tex + vec3(texel.x, 0.0, 0.0), 0.0, 1.0)).a;
    float fx0 = texture(field_tex, clamp(tex - vec3(texel.x, 0.0, 0.0), 0.0, 1.0)).a;
    float fy1 = texture(field_tex, clamp(tex + vec3(0.0, texel.y, 0.0), 0.0, 1.0)).a;
    float fy0 = texture(field_tex, clamp(tex - vec3(0.0, texel.y, 0.0), 0.0, 1.0)).a;
    float fz1 = texture(field_tex, clamp(tex + vec3(0.0, 0.0, texel.z), 0.0, 1.0)).a;
    float fz0 = texture(field_tex, clamp(tex - vec3(0.0, 0.0, texel.z), 0.0, 1.0)).a;

    vec3 grad_tex = vec3(fx1 - fx0, fy1 - fy0, fz1 - fz0) / (2.0 * texel);
    vec3 scale = volume_max - volume_min;
    return grad_tex / scale;
}

void emit_vertex(vec3 pos, vec4 field)
{
    world_pos = pos;
    vec3 grad = field_gradient(pos);
    if (length(grad) < 1e-5)
        grad = vec3(0.0, 1.0, 0.0);
    world_normal = normalize(-grad);
    albedo = field.rgb / max(field.a, 1e-6);
    gl_Position = projection * view * vec4(pos, 1.0);
    EmitVertex();
}

void emit_triangle(vec3 pos0, vec4 field0, vec3 pos1, vec4 field1, vec3 pos2, vec4 field2)
{
    emit_vertex(pos0, field0);
    emit_vertex(pos1, field1);
    emit_vertex(pos2, field2);
    EndPrimitive();
}

void interpolate(int a, int b, vec3 pos[4], vec4 field[4], out vec3 out_pos, out vec4 out_field)
{
    float v0 = field[a].a;
    float v1 = field[b].a;
    float denom = v1 - v0;
    float t = (abs(denom) > 1e-6) ? (iso_level - v0) / denom : 0.5;
    t = clamp(t, 0.0, 1.0);
    out_pos = mix(pos[a], pos[b], t);
    out_field = mix(field[a], field[b], t);
}

void emit_tetra(vec3 pos[4], vec4 field[4])
{
    bool inside[4];
    int in_list[4];
    int out_list[4];
    int in_count = 0;
    int out_count = 0;

    for (int i = 0; i < 4; ++i) {
        inside[i] = field[i].a >= iso_level;
        if (inside[i])
            in_list[in_count++] = i;
        else
            out_list[out_count++] = i;
    }

    if (in_count == 0 || in_count == 4)
        return;

    if (in_count == 1 || in_count == 3) {
        int v0 = (in_count == 1) ? in_list[0] : out_list[0];
        int o0 = (in_count == 1) ? out_list[0] : in_list[0];
        int o1 = (in_count == 1) ? out_list[1] : in_list[1];
        int o2 = (in_count == 1) ? out_list[2] : in_list[2];

        vec3 p0, p1, p2;
        vec4 f0, f1, f2;
        interpolate(v0, o0, pos, field, p0, f0);
        interpolate(v0, o1, pos, field, p1, f1);
        interpolate(v0, o2, pos, field, p2, f2);
        emit_triangle(p0, f0, p1, f1, p2, f2);
        return;
    }

    int i0 = in_list[0];
    int i1 = in_list[1];
    int o0 = out_list[0];
    int o1 = out_list[1];

    vec3 p00, p01, p10, p11;
    vec4 f00, f01, f10, f11;
    interpolate(i0, o0, pos, field, p00, f00);
    interpolate(i0, o1, pos, field, p01, f01);
    interpolate(i1, o0, pos, field, p10, f10);
    interpolate(i1, o1, pos, field, p11, f11);

    emit_triangle(p00, f00, p10, f10, p01, f01);
    emit_triangle(p01, f01, p10, f10, p11, f11);
}

void main()
{
    ivec3 base = cell[0];

    const ivec3 offsets[8] = ivec3[8](
        ivec3(0, 0, 0),
        ivec3(1, 0, 0),
        ivec3(1, 1, 0),
        ivec3(0, 1, 0),
        ivec3(0, 0, 1),
        ivec3(1, 0, 1),
        ivec3(1, 1, 1),
        ivec3(0, 1, 1)
    );

    vec3 corner_pos[8];
    vec4 corner_field[8];
    for (int i = 0; i < 8; ++i) {
        ivec3 ind = base + offsets[i];
        vec3 tex = (vec3(ind) + 0.5) / vec3(grid_size);
        corner_pos[i] = mix(volume_min, volume_max, tex);
        corner_field[i] = texelFetch(field_tex, ind, 0);
    }

    const ivec4 tetra_indices[6] = ivec4[6](
        ivec4(0, 5, 1, 6),
        ivec4(0, 1, 2, 6),
        ivec4(0, 2, 3, 6),
        ivec4(0, 3, 7, 6),
        ivec4(0, 7, 4, 6),
        ivec4(0, 4, 5, 6)
    );

    for (int t = 0; t < 6; ++t) {
        ivec4 ids = tetra_indices[t];
        vec3 pos[4];
        vec4 field[4];
        pos[0] = corner_pos[ids.x];
        pos[1] = corner_pos[ids.y];
        pos[2] = corner_pos[ids.z];
        pos[3] = corner_pos[ids.w];
        field[0] = corner_field[ids.x];
        field[1] = corner_field[ids.y];
        field[2] = corner_field[ids.z];
        field[3] = corner_field[ids.w];
        emit_tetra(pos, field);
    }
}
)";

const char fragment_shader_source[] = R"(#version 330 core
in vec3 world_pos;
in vec3 world_normal;
in vec3 albedo;

uniform vec3 camera_pos;
uniform vec3 sun_dir;
uniform vec3 sun_color;
uniform vec3 point_pos;
uniform vec3 point_color;
uniform vec3 ambient_color;
uniform float shininess;
uniform float specular_strength;

layout (location = 0) out vec4 out_color;

vec3 tonemap_uncharted2_raw(vec3 x)
{
    const float A = 0.15;
    const float B = 0.50;
    const float C = 0.10;
    const float D = 0.20;
    const float E = 0.02;
    const float F = 0.30;
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

vec3 tonemap_uncharted2(vec3 color)
{
    return tonemap_uncharted2_raw(color) / tonemap_uncharted2_raw(vec3(11.2));
}

void main()
{
    vec3 n = normalize(world_normal);
    vec3 v = normalize(camera_pos - world_pos);

    vec3 color = ambient_color * albedo;

    vec3 ld = normalize(sun_dir);
    float ndl = max(dot(n, ld), 0.0);
    vec3 h = normalize(v + ld);
    float spec = pow(max(dot(n, h), 0.0), shininess);
    color += sun_color * (albedo * ndl + specular_strength * spec);

    vec3 lvec = point_pos - world_pos;
    float dist = length(lvec);
    vec3 lp = (dist > 1e-4) ? (lvec / dist) : vec3(0.0, 1.0, 0.0);
    float atten = 1.0 / (1.0 + 0.6 * dist + 0.3 * dist * dist);
    float ndlp = max(dot(n, lp), 0.0);
    vec3 hp = normalize(v + lp);
    float specp = pow(max(dot(n, hp), 0.0), shininess);
    color += point_color * atten * (albedo * ndlp + specular_strength * specp);

    color = tonemap_uncharted2(color);
    color = pow(max(color, vec3(0.0)), vec3(1.0 / 2.2));
    out_color = vec4(color, 1.0);
}
)";

const char line_vertex_shader_source[] = R"(#version 330 core
layout (location = 0) in vec3 in_position;

uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * vec4(in_position, 1.0);
}
)";

const char line_fragment_shader_source[] = R"(#version 330 core
uniform vec3 line_color;

layout (location = 0) out vec4 out_color;

vec3 tonemap_uncharted2_raw(vec3 x)
{
    const float A = 0.15;
    const float B = 0.50;
    const float C = 0.10;
    const float D = 0.20;
    const float E = 0.02;
    const float F = 0.30;
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

vec3 tonemap_uncharted2(vec3 color)
{
    return tonemap_uncharted2_raw(color) / tonemap_uncharted2_raw(vec3(11.2));
}

void main()
{
    vec3 color = tonemap_uncharted2(line_color);
    color = pow(max(color, vec3(0.0)), vec3(1.0 / 2.2));
    out_color = vec4(color, 1.0);
}
)";

const char compute_shader_source[] = R"(#version 430 core
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

struct Ball {
    vec4 pos_radius;
    vec4 color;
};

layout(std430, binding = 0) buffer Balls
{
    Ball balls[];
};

layout(rgba32f, binding = 1) uniform image3D field_img;

uniform ivec3 grid_size;
uniform vec3 volume_min;
uniform vec3 volume_max;
uniform int ball_count;

void main()
{
    ivec3 id = ivec3(gl_GlobalInvocationID);
    if (any(greaterThanEqual(id, grid_size)))
        return;

    vec3 uvw = (vec3(id) + 0.5) / vec3(grid_size);
    vec3 pos = mix(volume_min, volume_max, uvw);

    vec3 color_sum = vec3(0.0);
    float weight_sum = 0.0;
    for (int i = 0; i < ball_count; ++i) {
        vec3 center = balls[i].pos_radius.xyz;
        float radius = balls[i].pos_radius.w;
        float dist = length(pos - center);
        float w = exp(-dist / max(radius, 1e-4));
        color_sum += w * balls[i].color.rgb;
        weight_sum += w;
    }

    imageStore(field_img, id, vec4(color_sum, weight_sum));
}
)";

struct BallData {
    glm::vec4 pos_radius;
    glm::vec4 color;
};

float hash01(int i, int seed) {
    float x = std::sin(float(i) * 12.9898f + float(seed) * 78.233f) * 43758.5453f;
    return x - std::floor(x);
}

int main() try {
    if (SDL_Init(SDL_INIT_VIDEO) != 0) sdl2_fail("SDL_Init: ");

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);

    SDL_Window* window = SDL_CreateWindow("Metaballs", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1280, 720,
                                          SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
    if (!window) sdl2_fail("SDL_CreateWindow: ");

    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    if (!gl_context) sdl2_fail("SDL_GL_CreateContext: ");

    SDL_GL_SetSwapInterval(1);

    if (auto result = glewInit(); result != GLEW_NO_ERROR) glew_fail("glewInit: ", result);

    if (!GLEW_VERSION_4_3) throw std::runtime_error("OpenGL 4.3 is required for compute shaders.");

    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glEnable(GL_MULTISAMPLE);

    GLuint vs = create_shader(GL_VERTEX_SHADER, vertex_shader_source);
    GLuint gs = create_shader(GL_GEOMETRY_SHADER, geometry_shader_source);
    GLuint fs = create_shader(GL_FRAGMENT_SHADER, fragment_shader_source);
    GLuint program = create_program({vs, gs, fs});
    glDeleteShader(vs);
    glDeleteShader(gs);
    glDeleteShader(fs);

    GLuint cs = create_shader(GL_COMPUTE_SHADER, compute_shader_source);
    GLuint compute_program = create_program({cs});
    glDeleteShader(cs);

    GLuint line_vs = create_shader(GL_VERTEX_SHADER, line_vertex_shader_source);
    GLuint line_fs = create_shader(GL_FRAGMENT_SHADER, line_fragment_shader_source);
    GLuint line_program = create_program({line_vs, line_fs});
    glDeleteShader(line_vs);
    glDeleteShader(line_fs);

    int grid_size = 72;
    const int grid_min = 24;
    const int grid_max = 384;
    const glm::vec3 volume_min(-3.6f, -3.6f, -3.6f);
    const glm::vec3 volume_max(3.6f, 3.6f, 3.6f);
    const glm::vec3 scene_extent = volume_max - volume_min;
    const glm::vec3 scene_margin = scene_extent * 0.12f;
    const glm::vec3 scene_wiggle = scene_extent * 0.04f;
    constexpr int grid_dim = 5;
    constexpr int ball_count = grid_dim * grid_dim * grid_dim;

    GLuint field_tex = 0;
    auto allocate_field_texture = [&]() {
        if (field_tex != 0) glDeleteTextures(1, &field_tex);
        glGenTextures(1, &field_tex);
        glBindTexture(GL_TEXTURE_3D, field_tex);
        glTexStorage3D(GL_TEXTURE_3D, 1, GL_RGBA32F, grid_size, grid_size, grid_size);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        glBindImageTexture(1, field_tex, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    };
    allocate_field_texture();

    glm::vec3 v000(volume_min.x, volume_min.y, volume_min.z);
    glm::vec3 v100(volume_max.x, volume_min.y, volume_min.z);
    glm::vec3 v110(volume_max.x, volume_max.y, volume_min.z);
    glm::vec3 v010(volume_min.x, volume_max.y, volume_min.z);
    glm::vec3 v001(volume_min.x, volume_min.y, volume_max.z);
    glm::vec3 v101(volume_max.x, volume_min.y, volume_max.z);
    glm::vec3 v111(volume_max.x, volume_max.y, volume_max.z);
    glm::vec3 v011(volume_min.x, volume_max.y, volume_max.z);

    const std::array<glm::vec3, 24> line_vertices = {v000, v100, v100, v110, v110, v010, v010, v000,
                                                     v001, v101, v101, v111, v111, v011, v011, v001,
                                                     v000, v001, v100, v101, v110, v111, v010, v011};

    GLuint line_vao = 0;
    GLuint line_vbo = 0;
    glGenVertexArrays(1, &line_vao);
    glBindVertexArray(line_vao);
    glGenBuffers(1, &line_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, line_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(line_vertices), line_vertices.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), reinterpret_cast<void*>(0));
    glBindVertexArray(0);

    GLuint dummy_vao = 0;
    glGenVertexArrays(1, &dummy_vao);
    glBindVertexArray(dummy_vao);

    GLuint ball_ssbo = 0;
    glGenBuffers(1, &ball_ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ball_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(BallData) * ball_count, nullptr, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ball_ssbo);

    GLuint cell_grid_location = glGetUniformLocation(program, "cell_grid");
    GLuint field_tex_location = glGetUniformLocation(program, "field_tex");
    GLuint iso_location = glGetUniformLocation(program, "iso_level");
    GLuint grid_location = glGetUniformLocation(program, "grid_size");
    GLuint volume_min_location = glGetUniformLocation(program, "volume_min");
    GLuint volume_max_location = glGetUniformLocation(program, "volume_max");
    GLuint view_location = glGetUniformLocation(program, "view");
    GLuint proj_location = glGetUniformLocation(program, "projection");
    GLuint camera_location = glGetUniformLocation(program, "camera_pos");
    GLuint sun_dir_location = glGetUniformLocation(program, "sun_dir");
    GLuint sun_color_location = glGetUniformLocation(program, "sun_color");
    GLuint point_pos_location = glGetUniformLocation(program, "point_pos");
    GLuint point_color_location = glGetUniformLocation(program, "point_color");
    GLuint ambient_location = glGetUniformLocation(program, "ambient_color");
    GLuint shininess_location = glGetUniformLocation(program, "shininess");
    GLuint specular_location = glGetUniformLocation(program, "specular_strength");

    GLuint line_view_location = glGetUniformLocation(line_program, "view");
    GLuint line_proj_location = glGetUniformLocation(line_program, "projection");
    GLuint line_color_location = glGetUniformLocation(line_program, "line_color");

    GLuint comp_grid_location = glGetUniformLocation(compute_program, "grid_size");
    GLuint comp_volume_min_location = glGetUniformLocation(compute_program, "volume_min");
    GLuint comp_volume_max_location = glGetUniformLocation(compute_program, "volume_max");
    GLuint comp_ball_count_location = glGetUniformLocation(compute_program, "ball_count");

    int width = 0;
    int height = 0;
    SDL_GetWindowSize(window, &width, &height);
    glViewport(0, 0, width, height);

    const float iso_min = 0.1f;
    const float iso_max = 0.7f;
    const float iso_step = 0.05f;
    float iso_ratio = 0.5f;
    auto change_grid_detail = [&](int delta) {
        int new_size = std::clamp(grid_size + delta, grid_min, grid_max);
        if (new_size != grid_size) {
            grid_size = new_size;
            allocate_field_texture();
        }
    };

    SDL_SetRelativeMouseMode(SDL_TRUE);
    std::array<bool, SDL_NUM_SCANCODES> key_down{};
    auto last_frame_start = std::chrono::steady_clock::now();
    float time = 0.f;

    glm::vec3 cam_pos(0.f, 0.3f, 3.f);
    glm::vec3 init_forward = glm::normalize(-cam_pos);
    float yaw = std::atan2(init_forward.x, init_forward.z);
    float pitch = std::asin(init_forward.y);

    bool running = true;
    while (running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) running = false;
            if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
                width = event.window.data1;
                height = event.window.data2;
                glViewport(0, 0, width, height);
            }
            if (event.type == SDL_KEYDOWN) {
                key_down[event.key.keysym.scancode] = true;
                if (event.key.keysym.sym == SDLK_ESCAPE) running = false;
                if (event.key.keysym.sym == SDLK_UP) iso_ratio = std::min(iso_max, iso_ratio + iso_step);
                if (event.key.keysym.sym == SDLK_DOWN) iso_ratio = std::max(iso_min, iso_ratio - iso_step);
                if (event.key.keysym.sym == SDLK_RIGHT) change_grid_detail(4);
                if (event.key.keysym.sym == SDLK_LEFT) change_grid_detail(-4);
            }
            if (event.type == SDL_KEYUP) {
                key_down[event.key.keysym.scancode] = false;
            }
            if (event.type == SDL_MOUSEMOTION) {
                const float sens = 0.0025f;
                yaw -= sens * float(event.motion.xrel);
                pitch -= sens * float(event.motion.yrel);
                const float lim = glm::radians(89.f);
                pitch = std::clamp(pitch, -lim, lim);
            }
        }

        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration_cast<std::chrono::duration<float>>(now - last_frame_start).count();
        last_frame_start = now;
        time += dt;

        std::array<BallData, ball_count> balls{};
        for (int i = 0; i < ball_count; ++i) {
            int ix = i % grid_dim;
            int iy = (i / grid_dim) % grid_dim;
            int iz = i / (grid_dim * grid_dim);
            glm::vec3 cell = glm::vec3(ix, iy, iz) / float(grid_dim - 1);
            glm::vec3 base = volume_min + scene_margin + cell * (scene_extent - 2.f * scene_margin);
            glm::vec3 jitter(hash01(i, 1) - 0.5f, hash01(i, 2) - 0.5f, hash01(i, 3) - 0.5f);
            base += jitter * scene_extent * 0.03f;
            glm::vec3 wiggle(std::sin(time * 0.35f + float(i) * 1.7f), std::sin(time * 0.5f + float(i) * 2.3f),
                             std::sin(time * 0.65f + float(i) * 1.1f));
            glm::vec3 pos = base + wiggle * scene_wiggle;

            float radius_min = 0.018f * scene_extent.x;
            float radius_max = 0.05f * scene_extent.x;
            float t = hash01(i, 5);
            float radius = radius_min + (radius_max - radius_min) * std::pow(t, 0.6f);
            float chaos = hash01(i, 7);
            if (chaos > 0.92f)
                radius *= 2.2f;
            else if (chaos > 0.82f)
                radius *= 1.6f;

            float hue = float(i) / float(ball_count);
            glm::vec3 color(0.5f + 0.5f * std::cos(6.28318f * (hue + 0.f)),
                            0.5f + 0.5f * std::cos(6.28318f * (hue + 0.33f)),
                            0.5f + 0.5f * std::cos(6.28318f * (hue + 0.67f)));

            balls[i].pos_radius = glm::vec4(pos, radius);
            balls[i].color = glm::vec4(color, 1.f);
        }

        float max_weight = 0.f;
        for (int i = 0; i < ball_count; ++i) {
            float sum = 0.f;
            const glm::vec3 pi = glm::vec3(balls[i].pos_radius);
            for (int j = 0; j < ball_count; ++j) {
                const glm::vec3 pj = glm::vec3(balls[j].pos_radius);
                float radius = balls[j].pos_radius.w;
                float dist = glm::length(pi - pj);
                float w = std::exp(-dist / std::max(radius, 1e-4f));
                sum += w;
            }
            max_weight = std::max(max_weight, sum);
        }
        max_weight = std::max(max_weight, 1e-4f);
        float iso_threshold = iso_ratio * max_weight;

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ball_ssbo);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(BallData) * ball_count, balls.data());

        glUseProgram(compute_program);
        glUniform3i(comp_grid_location, grid_size, grid_size, grid_size);
        glUniform3f(comp_volume_min_location, volume_min.x, volume_min.y, volume_min.z);
        glUniform3f(comp_volume_max_location, volume_max.x, volume_max.y, volume_max.z);
        glUniform1i(comp_ball_count_location, ball_count);

        const int group_size = 4;
        const int gx = (grid_size + group_size - 1) / group_size;
        const int gy = (grid_size + group_size - 1) / group_size;
        const int gz = (grid_size + group_size - 1) / group_size;
        glDispatchCompute(gx, gy, gz);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

        glClearColor(0.05f, 0.06f, 0.08f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glm::vec3 forward = glm::normalize(
            glm::vec3(std::cos(pitch) * std::sin(yaw), std::sin(pitch), std::cos(pitch) * std::cos(yaw)));
        glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0.f, 1.f, 0.f)));
        glm::vec3 up = glm::normalize(glm::cross(right, forward));

        float speed = key_down[SDL_SCANCODE_LSHIFT] ? 2.5f : 1.f;
        float step = speed * dt;
        if (key_down[SDL_SCANCODE_W]) cam_pos += forward * step;
        if (key_down[SDL_SCANCODE_S]) cam_pos -= forward * step;
        if (key_down[SDL_SCANCODE_D]) cam_pos += right * step;
        if (key_down[SDL_SCANCODE_A]) cam_pos -= right * step;
        if (key_down[SDL_SCANCODE_E]) cam_pos += up * step;
        if (key_down[SDL_SCANCODE_Q]) cam_pos -= up * step;

        float aspect = (height > 0) ? float(width) / float(height) : 1.f;
        glm::mat4 view = glm::lookAt(cam_pos, cam_pos + forward, glm::vec3(0.f, 1.f, 0.f));
        glm::mat4 projection = glm::perspective(glm::radians(45.f), aspect, 0.1f, 20.f);

        glm::vec3 sun_dir = glm::normalize(glm::vec3(std::cos(time * 0.4f), 0.7f, std::sin(time * 0.4f)));
        glm::vec3 sun_color(1.4f, 1.2f, 1.f);
        glm::vec3 point_pos(std::cos(time * 0.9f) * 1.8f, 0.6f + std::sin(time * 0.7f) * 0.4f,
                            std::sin(time * 0.9f) * 1.8f);
        glm::vec3 point_color(0.8f, 0.9f, 1.1f);

        glUseProgram(program);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, field_tex);
        glUniform1i(field_tex_location, 0);
        glUniform1f(iso_location, iso_threshold);
        glUniform3i(grid_location, grid_size, grid_size, grid_size);
        glUniform3i(cell_grid_location, grid_size - 1, grid_size - 1, grid_size - 1);
        glUniform3f(volume_min_location, volume_min.x, volume_min.y, volume_min.z);
        glUniform3f(volume_max_location, volume_max.x, volume_max.y, volume_max.z);
        glUniformMatrix4fv(view_location, 1, GL_FALSE, reinterpret_cast<float*>(&view));
        glUniformMatrix4fv(proj_location, 1, GL_FALSE, reinterpret_cast<float*>(&projection));
        glUniform3f(camera_location, cam_pos.x, cam_pos.y, cam_pos.z);
        glUniform3f(sun_dir_location, sun_dir.x, sun_dir.y, sun_dir.z);
        glUniform3f(sun_color_location, sun_color.x, sun_color.y, sun_color.z);
        glUniform3f(point_pos_location, point_pos.x, point_pos.y, point_pos.z);
        glUniform3f(point_color_location, point_color.x, point_color.y, point_color.z);
        glUniform3f(ambient_location, 0.08f, 0.08f, 0.1f);
        glUniform1f(shininess_location, 64.f);
        glUniform1f(specular_location, 0.6f);

        int cell_count = (grid_size - 1) * (grid_size - 1) * (grid_size - 1);
        glBindVertexArray(dummy_vao);
        glDrawArrays(GL_POINTS, 0, cell_count);

        glUseProgram(line_program);
        glUniformMatrix4fv(line_view_location, 1, GL_FALSE, reinterpret_cast<float*>(&view));
        glUniformMatrix4fv(line_proj_location, 1, GL_FALSE, reinterpret_cast<float*>(&projection));
        glUniform3f(line_color_location, 1.f, 1.f, 1.f);
        glBindVertexArray(line_vao);
        glLineWidth(3.f);
        glDrawArrays(GL_LINES, 0, line_vertices.size());

        SDL_GL_SwapWindow(window);
    }

    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(window);
    SDL_Quit();
} catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
}
