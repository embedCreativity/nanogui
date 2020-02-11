/*
    NanoGUI was developed by Wenzel Jakob <wenzel.jakob@epfl.ch>.
    The widget drawing code is based on the NanoVG demo application
    by Mikko Mononen.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

/**
 * \file nanogui/shader.h
 *
 * \brief Defines abstractions for shaders that work with OpenGL,
 * OpenGL ES, and Metal.
 */

#pragma once

#include <nanogui/object.h>
#include <unordered_map>

NAMESPACE_BEGIN(nanogui)

enum class VariableType { Invalid = 0, Int8, UInt8, Int16, UInt16,
                          Int32, UInt32, Int64, UInt64, Float16,
                          Float32, Float64, Bool };

template <typename T> constexpr VariableType get_type() {
    if constexpr (std::is_same_v<T, bool>)
        return VariableType::Bool;

    if constexpr (is_integral_v<T>) {
        if constexpr (sizeof(T) == 1)
            return std::is_signed_v<T> ? VariableType::Int8 : VariableType::UInt8;
        else if constexpr (sizeof(T) == 2)
            return std::is_signed_v<T> ? VariableType::Int16 : VariableType::UInt16;
        else if constexpr (sizeof(T) == 4)
            return std::is_signed_v<T> ? VariableType::Int32 : VariableType::UInt32;
        else if constexpr (sizeof(T) == 8)
            return std::is_signed_v<T> ? VariableType::Int64 : VariableType::UInt64;
    } else if constexpr (std::is_floating_point_v<T>) {
        if constexpr (sizeof(T) == 2)
            return VariableType::Float16;
        else if constexpr (sizeof(T) == 4)
            return VariableType::Float32;
        else if constexpr (sizeof(T) == 8)
            return VariableType::Float64;
    }

    return VariableType::Invalid;
}

/// Return the size in bytes associated with a specific variable type
extern NANOGUI_EXPORT size_t type_size(VariableType type);

/// Return the name (e.g. "uint8") associated with a specific variable type
extern NANOGUI_EXPORT const char *type_name(VariableType type);

class NANOGUI_EXPORT Shader : public Object {
public:
    /// The type of geometry that should be rendered
    enum class PrimitiveType {
        Point,
        Line,
        LineStrip,
        Triangle,
        TriangleStrip
    };

    /// Alpha blending mode
    enum class BlendMode {
        None,
        AlphaBlend // alpha * new_color + (1 - alpha) * old_color
    };

    /**
     * \brief Initialize the shader using the specified source strings.
     *
     * \param render_pass
     *     RenderPass object encoding targets to which color, depth,
     *     and stencil information will be rendered.
     *
     * \param name
     *     A name identifying this shader
     *
     * \param vertex_shader
     *     The source of the vertex shader as a string.
     *
     * \param fragment_shader
     *     The source of the fragment shader as a string.
     */
    Shader(RenderPass *render_pass,
           const std::string &name,
           const std::string &vertex_shader,
           const std::string &fragment_shader,
           BlendMode blend_mode = BlendMode::None);

    /// Return the render pass associated with this shader
    RenderPass *render_pass() { return m_render_pass; }

    /// Return the name of this shader
    const std::string &name() const { return m_name; }

    /// Return the blending mode of this shader
    BlendMode blend_mode() const { return m_blend_mode; }

    /**
     * \brief Upload a buffer (e.g. vertex positions) that will be associated
     * with a named shader parameter.
     *
     * Note that this function should be used both for 'varying' and 'uniform'
     * data---the implementation takes care of routing the data to the right
     * endpoint. Matrices should be specified in column-major order.
     *
     * The buffer will be replaced if it is already present.
     */
    void set_buffer(const std::string &name, VariableType type, size_t ndim,
                    std::array<size_t, 3> shape, const void *data);

    /**
     * \brief Upload a uniform variable (e.g. a vector or matrix) that will be
     * associated with a named shader parameter.
     */
    template <typename Array> void set_uniform(const std::string &name,
                                               const Array &value) {
        std::array<size_t, 3> shape = { 1, 1, 1 };
        size_t ndim = (size_t) -1;
        const void *data;
        VariableType vtype;

        if constexpr (std::is_scalar_v<Array>) {
            data = &value;
            ndim = 0;
            vtype = get_type<Array>();
        } else if constexpr (IsBuiltinArray) {
            ndim = 1;
            shape[0] = Array::Size;
            vtype = get_type<typename Array::Value>();
        } else if constexpr (IsEnokiArray) {
            if constexpr (Array::Depth == 1) {
                shape[0] = value.size();
                ndim = 1;
            } else if constexpr (Array::Depth == 2) {
                shape[0] = value.size();
                shape[1] = value[0].size();
                ndim = 2;
            } else if constexpr (Array::Depth == 3) {
                shape[0] = value.size();
                shape[1] = value[0].size();
                shape[2] = value[0][0].size();
                ndim = 3;
            }
            data = value.data();
            vtype = get_type<typename Array::Scalar>();
        }

        if (ndim == (size_t) -1)
            throw std::runtime_error("Shader::set_uniform(): invalid input array dimension!");

        set_buffer(name, vtype, ndim, shape, data);
    }

    /**
     * \brief Associate a texture with a named shader parameter
     *
     * The association will be replaced if it is already present.
     */
    void set_texture(const std::string &name, Texture *texture);

    /**
     * \brief Begin drawing using this shader
     *
     * Note that any updates to 'uniform' and 'varying' shader parameters
     * *must* occur prior to this method call.
     *
     * The Python bindings also include extra \c __enter__ and \c __exit__
     * aliases so that the shader can be activated via Pythons 'with'
     * statement.
     */
    void begin();

    /// End drawing using this shader
    void end();

    /**
     * \brief Render geometry arrays, either directly or
     * using an index array.
     *
     * \param primitive_type
     *     What type of geometry should be rendered?
     *
     * \param offset
     *     First index to render. Must be a multiple of 2 or 3 for lines and
     *     triangles, respectively (unless specified using strips).
     *
     * \param offset
     *     Number of indices to render. Must be a multiple of 2 or 3 for lines
     *     and triangles, respectively (unless specified using strips).
     *
     * \param indexed
     *     Render indexed geometry? In this case, an
     *     \c uint32_t valued buffer with name \c indices
     *     must have been uploaded using \ref set().
     */
    void draw_array(PrimitiveType primitive_type,
                    size_t offset, size_t count,
                    bool indexed = false);

#if defined(NANOGUI_USE_OPENGL) || defined(NANOGUI_USE_GLES)
    uint32_t shader_handle() const { return m_shader_handle; }
#elif defined(NANOGUI_USE_METAL)
    void *pipeline_state() const { return m_pipeline_state; }
#endif

#if defined(NANOGUI_USE_OPENGL)
    uint32_t vertex_array_handle() const { return m_vertex_array_handle; }
#endif

protected:
    enum BufferType {
        Unknown = 0,
        VertexBuffer,
        VertexTexture,
        VertexSampler,
        FragmentBuffer,
        FragmentTexture,
        FragmentSampler,
        UniformBuffer,
        IndexBuffer,
    };

    struct Buffer {
        void *buffer = nullptr;
        BufferType type = Unknown;
        VariableType dtype = VariableType::Invalid;
        int index = 0;
        size_t ndim = 0;
        std::array<size_t, 3> shape { 0, 0, 0 };
        size_t size = 0;
        bool dirty = false;

        std::string to_string() const;
    };

    /// Release all resources
    virtual ~Shader();

protected:
    RenderPass* m_render_pass;
    std::string m_name;
    std::unordered_map<std::string, Buffer> m_buffers;
    BlendMode m_blend_mode;

    #if defined(NANOGUI_USE_OPENGL) || defined(NANOGUI_USE_GLES)
        uint32_t m_shader_handle = 0;
    #  if defined(NANOGUI_USE_OPENGL)
        uint32_t m_vertex_array_handle = 0;
        bool m_uses_point_size = false;
    #  endif
    #elif defined(NANOGUI_USE_METAL)
        void *m_pipeline_state;
    #endif
};

/// Access binary data stored in nanogui_resources.cpp
#define NANOGUI_RESOURCE_STRING(name) std::string(name, name + name##_size)

/// Access a shader stored in nanogui_resources.cpp
#if defined(NANOGUI_USE_OPENGL)
#  define NANOGUI_SHADER(name) NANOGUI_RESOURCE_STRING(name##_gl)
#elif defined(NANOGUI_USE_GLES)
#  define NANOGUI_SHADER(name) NANOGUI_RESOURCE_STRING(name##_gles)
#elif defined(NANOGUI_USE_METAL)
#  define NANOGUI_SHADER(name) NANOGUI_RESOURCE_STRING(name##_metallib)
#endif


NAMESPACE_END(nanogui)
