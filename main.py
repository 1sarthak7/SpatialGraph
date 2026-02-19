import sys
import cv2
import pygame
import moderngl
import mediapipe as mp
import numpy as np
import glm

# ==========================================
# SHADERS (GLSL 330 Core)
# ==========================================
WEBCAM_VERT = """#version 330 core
in vec2 in_vert; in vec2 in_tex; out vec2 v_tex;
void main() { gl_Position = vec4(in_vert, 0.0, 1.0); v_tex = in_tex; }"""

WEBCAM_FRAG = """#version 330 core
uniform sampler2D tex; in vec2 v_tex; out vec4 f_color;
void main() { f_color = texture(tex, v_tex); }"""

BRUSH_VERT = """#version 330 core
in vec3 in_pos; in vec4 in_color;
uniform mat4 vp; out vec4 v_color;
void main() {
    v_color = in_color;
    gl_Position = vp * vec4(in_pos, 1.0);
    gl_Position.z -= 0.001; // Mitigate z-fighting
}"""

BRUSH_FRAG = """#version 330 core
in vec4 v_color; out vec4 f_color;
void main() { f_color = v_color; }"""

COMPOSITE_VERT = """#version 330 core
in vec2 in_vert; in vec2 in_tex; out vec2 uv;
void main() { gl_Position = vec4(in_vert, 0.0, 1.0); uv = in_tex; }"""

COMPOSITE_FRAG = """#version 330 core
uniform sampler2D webcam_tex;
uniform sampler2D scene_tex; 
uniform sampler2D bloom_tex;
in vec2 uv; out vec4 f_color;

void main() {
    vec3 webcam = texture(webcam_tex, uv).rgb;
    vec4 ar_brush = texture(scene_tex, uv); // Contains color (rgb) and transparency (a)
    vec3 ar_bloom = texture(bloom_tex, uv).rgb;
    
    // 1. Alpha-blend the solid brush over the real-world webcam feed
    vec3 base_composite = mix(webcam, ar_brush.rgb, ar_brush.a);
    
    // 2. Add the glowing bloom purely as light (additive blending)
    vec3 final_result = base_composite + (ar_bloom * 1.5);
    
    f_color = vec4(final_result, 1.0); 
}"""

BRIGHT_PASS_FRAG = """#version 330 core
uniform sampler2D tex;
uniform float threshold;
in vec2 uv; out vec4 f_color;
void main() {
    vec4 color = texture(tex, uv);
    // Calculate relative luminance
    float brightness = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));
    if(brightness > threshold) f_color = vec4(color.rgb, 1.0);
    else f_color = vec4(0.0, 0.0, 0.0, 1.0);
}"""

BLUR_FRAG = """#version 330 core
uniform sampler2D tex;
uniform vec2 direction;
in vec2 uv; out vec4 f_color;
void main() {
    vec2 off = 1.0 / textureSize(tex, 0);
    vec3 color = texture(tex, uv).rgb * 0.227027;
    color += texture(tex, uv + (direction * off * 1.0)).rgb * 0.1945946;
    color += texture(tex, uv - (direction * off * 1.0)).rgb * 0.1945946;
    color += texture(tex, uv + (direction * off * 2.0)).rgb * 0.1216216;
    color += texture(tex, uv - (direction * off * 2.0)).rgb * 0.1216216;
    color += texture(tex, uv + (direction * off * 3.0)).rgb * 0.054054;
    color += texture(tex, uv - (direction * off * 3.0)).rgb * 0.054054;
    f_color = vec4(color, 1.0);
}"""

# ==========================================
# CORE SYSTEMS
# ==========================================
class SpatialFilter:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.val = None

    def update(self, new_val):
        if self.val is None:
            self.val = np.array(new_val, dtype='f4')
        else:
            self.val = self.alpha * np.array(new_val, dtype='f4') + (1.0 - self.alpha) * self.val
        return self.val

class Camera:
    def __init__(self, width, height):
        self.aspect = width / height
        self.fov, self.near, self.far = 60.0, 0.1, 100.0
        self.position = glm.vec3(0, 0, 0)
        self.target = glm.vec3(0, 0, -2) 
        self.up = glm.vec3(0, 1, 0)
        self.update_matrices()

    def update_matrices(self):
        self.view = glm.lookAt(self.position, self.target, self.up)
        self.proj = glm.perspective(glm.radians(self.fov), self.aspect, self.near, self.far)
        self.vp = self.proj * self.view

class BrushSystem:
    def __init__(self, ctx, max_points=5000):
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=BRUSH_VERT, fragment_shader=BRUSH_FRAG)
        self.max_verts = max_points * 2
        
        self.vbo = ctx.buffer(reserve=self.max_verts * 7 * 4)
        self.vao = ctx.vertex_array(self.prog, [(self.vbo, '3f 4f', 'in_pos', 'in_color')])
        
        self.vertices = np.zeros((self.max_verts, 7), dtype='f4')
        self.v_count = 0
        self.last_pos = None

    def add_point(self, pos, color=(0.0, 1.0, 0.8)):
        if self.v_count >= self.max_verts - 2: return
        if self.last_pos is None:
            self.last_pos = pos
            return

        dir_vec = pos - self.last_pos
        velocity = np.linalg.norm(dir_vec)
        if velocity < 0.001: return

        width = np.clip(0.2 / (velocity * 10.0 + 1.0), 0.05, 0.2) 
        forward = dir_vec / velocity
        up = np.array([0, 1, 0])
        right = np.cross(forward, up)
        if np.linalg.norm(right) < 0.001: right = np.array([1, 0, 0])
            
        p1 = pos + right * width
        p2 = pos - right * width
        
        c = [*color, 0.9] 
        self.vertices[self.v_count] = [*p1, *c]
        self.vertices[self.v_count+1] = [*p2, *c]
        self.v_count += 2
        self.last_pos = pos

    def render(self, vp_matrix_bytes):
        if self.v_count > 2:
            self.vbo.write(self.vertices[:self.v_count].tobytes())
            self.prog['vp'].write(vp_matrix_bytes)
            self.ctx.enable(self.ctx.BLEND | self.ctx.DEPTH_TEST)
            self.ctx.blend_func = self.ctx.SRC_ALPHA, self.ctx.ONE_MINUS_SRC_ALPHA
            self.vao.render(mode=self.ctx.TRIANGLE_STRIP, vertices=self.v_count)

# ==========================================
# ENGINE APPLICATION
# ==========================================
class AREngine:
    def __init__(self, width=1280, height=720):
        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        
        self.screen = pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
        self.ctx = moderngl.create_context()
        self.clock = pygame.time.Clock()

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.mp_hands = mp.solutions.hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.filter = SpatialFilter(alpha=0.25) 

        self.camera = Camera(width, height)
        self.brush = BrushSystem(self.ctx)

        self.quad_vbo = self.ctx.buffer(np.array([
            -1.0, -1.0, 0.0, 1.0,  1.0, -1.0, 1.0, 1.0,
            -1.0,  1.0, 0.0, 0.0,  1.0,  1.0, 1.0, 0.0,
        ], dtype='f4'))
        
        self.webcam_prog = self.ctx.program(vertex_shader=WEBCAM_VERT, fragment_shader=WEBCAM_FRAG)
        self.webcam_vao = self.ctx.vertex_array(self.webcam_prog, [(self.quad_vbo, '2f 2f', 'in_vert', 'in_tex')])
        self.webcam_tex = self.ctx.texture((width, height), 3)

        self.comp_prog = self.ctx.program(vertex_shader=COMPOSITE_VERT, fragment_shader=COMPOSITE_FRAG)
        self.comp_vao = self.ctx.vertex_array(self.comp_prog, [(self.quad_vbo, '2f 2f', 'in_vert', 'in_tex')])
        
        # Post-Processing FBOs
        self.scene_tex = self.ctx.texture((width, height), 4, dtype='f2') 
        self.depth_rb = self.ctx.depth_renderbuffer((width, height))
        self.scene_fbo = self.ctx.framebuffer(self.scene_tex, self.depth_rb)

        self.ping_tex = self.ctx.texture((width, height), 4, dtype='f2')
        self.pong_tex = self.ctx.texture((width, height), 4, dtype='f2')
        self.ping_fbo = self.ctx.framebuffer(self.ping_tex)
        self.pong_fbo = self.ctx.framebuffer(self.pong_tex)

        self.bright_prog = self.ctx.program(vertex_shader=COMPOSITE_VERT, fragment_shader=BRIGHT_PASS_FRAG)
        self.bright_vao = self.ctx.vertex_array(self.bright_prog, [(self.quad_vbo, '2f 2f', 'in_vert', 'in_tex')])

        self.blur_prog = self.ctx.program(vertex_shader=COMPOSITE_VERT, fragment_shader=BLUR_FRAG)
        self.blur_vao = self.ctx.vertex_array(self.blur_prog, [(self.quad_vbo, '2f 2f', 'in_vert', 'in_tex')])

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret: return None
        
        frame = cv2.flip(frame, 1) 
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.mp_hands.process(rgb_frame)
        self.webcam_tex.write(rgb_frame.tobytes())
        
        return results

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.brush.v_count = 0 

            results = self.process_frame()

            if results and results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                idx_tip = hand.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                
                aspect = self.camera.aspect
                raw_x = (idx_tip.x - 0.5) * 5.0 * aspect 
                raw_y = -(idx_tip.y - 0.5) * 5.0 
                raw_z = -3.0 
                
                smooth_pos = self.filter.update([raw_x, raw_y, raw_z])
                self.brush.add_point(smooth_pos)
            else:
                self.brush.last_pos = None 

            # --- RENDER PASS 1: AR Scene FBO (BRUSH ONLY) ---
            self.scene_fbo.use()
            # Clear to pure transparent black! (R, G, B, Alpha)
            self.ctx.clear(0.0, 0.0, 0.0, 0.0) 
            self.ctx.disable(self.ctx.DEPTH_TEST)
            
            # Note: We NO LONGER render the webcam here. Only the AR brush.
            self.brush.render(self.camera.vp.to_bytes())

            # --- RENDER PASS 2: Extract Bright Pixels ---
            # (This now only processes the AR brush, ignoring the real world)
            self.ping_fbo.use()
            self.ctx.clear(0.0, 0.0, 0.0, 1.0)
            self.bright_prog['tex'].value = 0
            self.bright_prog['threshold'].value = 0.5 
            self.scene_tex.use(0)
            self.bright_vao.render(mode=self.ctx.TRIANGLE_STRIP)

            # --- RENDER PASS 3: Ping-Pong Gaussian Blur ---
            self.blur_prog['tex'].value = 0
            blur_iterations = 6 
            for _ in range(blur_iterations):
                # Horizontal blur
                self.pong_fbo.use()
                self.blur_prog['direction'].value = (1.0, 0.0)
                self.ping_tex.use(0)
                self.blur_vao.render(mode=self.ctx.TRIANGLE_STRIP)

                # Vertical blur
                self.ping_fbo.use()
                self.blur_prog['direction'].value = (0.0, 1.0)
                self.pong_tex.use(0)
                self.blur_vao.render(mode=self.ctx.TRIANGLE_STRIP)

            # --- RENDER PASS 4: Final Composite to Screen ---
            self.ctx.screen.use()
            self.ctx.clear()
            self.ctx.disable(self.ctx.DEPTH_TEST)
            
            # Bind all THREE textures to different Texture Units
            self.comp_prog['webcam_tex'].value = 0
            self.comp_prog['scene_tex'].value = 1
            self.comp_prog['bloom_tex'].value = 2
            
            self.webcam_tex.use(0)  # Unit 0: Real world
            self.scene_tex.use(1)   # Unit 1: Solid AR Brush
            self.ping_tex.use(2)    # Unit 2: Blurred AR Glow
            
            self.comp_vao.render(mode=self.ctx.TRIANGLE_STRIP)

            pygame.display.flip()
            self.clock.tick(60)
            pygame.display.set_caption(f"AR Engine - FPS: {self.clock.get_fps():.1f}")

        self.cap.release()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    app = AREngine()
    app.run()