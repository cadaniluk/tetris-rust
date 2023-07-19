use std::time::{Instant, Duration};
use glfw::{Action, Context, Key};
use gl;
use nalgebra_glm::{TMat4, ortho};
use std::process;
use std::ptr;
use std::mem;
use rand;
use rand::Rng;

#[macro_use]
extern crate lazy_static;

enum RotateDir {
    Left, Right
}

const DESCENT_TIMEOUT: Duration = Duration::from_millis(500);
const MOVE_TIMEOUT: Duration = Duration::from_millis(75);

const VERT_SHADER: &str = r#"
    #version 430 core

    layout(location = 0) in uvec2 base_pos;
    layout(location = 1) in vec4 vert_color;
    out vec4 frag_color;

    layout(location = 0) uniform mat4 ortho_proj;
    layout(location = 1) uniform uvec2 offset;

    void main() {
        vec2 pos = base_pos + offset;
        pos.x *= 10.;
        pos.y *= 5.;

        gl_Position = ortho_proj * vec4(pos, 0., 1.);
        frag_color = vert_color;
    }
"#;

const FRAG_SHADER: &str = r#"
    #version 430 core

    in vec4 frag_color;
    out vec4 out_color;

    void main() {
        out_color = frag_color;
    }
"#;

fn handle_event(win: &mut glfw::Window, event: glfw::WindowEvent) -> Option<(Key, Action)> {
    match event {
        glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) => win.set_should_close(true),
        glfw::WindowEvent::Key(key, _, action, _) => {
            return Some((key, action))
        },
        _ => {}
    }

    None
}

const WORLD_WIDTH: u32 = 100 / WORLD_SQUARE_WIDTH;  // in square widths
const WORLD_HEIGHT: u32 = 100 / WORLD_SQUARE_HEIGHT;    // in square heights
const WORLD_SQUARE_WIDTH: u32 = 10;
const WORLD_SQUARE_HEIGHT: u32 = 5;
const WIN_WIDTH: u32 = 400;
const WIN_HEIGHT: u32 = 800;

const COLOR_BLUE: [f32; 4] = [0., 0.75, 1., 1.];
const COLOR_YELLOW: [f32; 4] = [1., 1., 0., 1.];
const COLOR_PURPLE: [f32; 4] = [1., 0., 1., 1.];
const COLOR_ORANGE: [f32; 4] = [1., 0.65, 0., 1.];
const COLOR_GREEN: [f32; 4] = [0., 1., 0., 1.];

lazy_static! {
    static ref ORTHO_PROJ: TMat4<f32> = ortho(0., (WORLD_WIDTH * WORLD_SQUARE_WIDTH) as f32, 0., (WORLD_HEIGHT * WORLD_SQUARE_HEIGHT) as f32 , -1., 1.);
    static ref STRAIGHT_SHAPE: [Vec<[u32; 2]>; 4] = [vec![[0, 0], [1, 0], [2, 0], [3, 0]], vec![[0, 0], [0, 1], [0, 2], [0, 3]], vec![[0, 0], [1, 0], [2, 0], [3, 0]], vec![[0, 0], [0, 1], [0, 2], [0, 3]]];
    static ref SQUARE_SHAPE: [Vec<[u32; 2]>; 4] = [vec![[0, 0], [0, 1], [1, 0], [1, 1]], vec![[0, 0], [0, 1], [1, 0], [1, 1]], vec![[0, 0], [0, 1], [1, 0], [1, 1]], vec![[0, 0], [0, 1], [1, 0], [1, 1]]];
    static ref L_SHAPE: [Vec<[u32; 2]>; 4] = [vec![[0, 0], [0, 1], [0, 2], [1, 0]], vec![[0, 0], [1, 0], [2, 0], [2, 1]], vec![[1, 0], [1, 1], [1, 2], [0, 2]], vec![[0, 0], [0, 1], [1, 1], [2, 1]]];
    static ref T_SHAPE: [Vec<[u32; 2]>; 4] = [vec![[0, 1], [1, 1], [2, 1], [1, 0]], vec![[0, 0], [0, 1], [0, 2], [1, 1]], vec![[0, 0], [1, 0], [2, 0], [1, 1]], vec![[0, 1], [1, 0], [1, 1], [1, 2]]];
    static ref S_SHAPE: [Vec<[u32; 2]>; 4] = [vec![[0, 0], [1, 0], [1, 1], [2, 1]], vec![[0, 1], [0, 2], [1, 1], [1, 0]], vec![[0, 0], [1, 0], [1, 1], [2, 1]], vec![[0, 1], [0, 2], [1, 1], [1, 0]]];
}

fn square_at(pos: (u32, u32)) -> [[u32; 2]; 6] {
    [[pos.0, pos.1 + 1], [pos.0, pos.1], [pos.0 + 1, pos.1], [pos.0, pos.1 + 1], [pos.0 + 1, pos.1], [pos.0 + 1, pos.1 + 1]]
}

struct Piece<'a> {
    shader_prog: u32,
    vao: u32,
    vert_vbo: u32,
    colors_vbo: u32,
    rotations: &'a [Vec<[u32; 2]>; 4],
    vertices: [Vec<[u32; 2]>; 4],
    pos_changes: [[i32; 2]; 4],
    curr_rotation: usize,
    pos: [u32; 2],
    width: u32,
    height: u32,
    color: [f32; 4]
}

impl<'a> Piece<'a> {
    pub fn new(shader_prog: u32, x_init: u32, rotations: &'a [Vec<[u32; 2]>; 4], pos_changes: [[i32; 2]; 4], color: [f32; 4]) -> Self {
        let mut vao = 0;
        let mut vert_vbo = 0;
        let mut colors_vbo = 0;

        let vertices: [Vec<[u32; 2]>; 4] = [
            rotations[0].iter().map(|pos| square_at((pos[0], pos[1]))).collect::<Vec<_>>().concat(),
            rotations[1].iter().map(|pos| square_at((pos[0], pos[1]))).collect::<Vec<_>>().concat(),
            rotations[2].iter().map(|pos| square_at((pos[0], pos[1]))).collect::<Vec<_>>().concat(),
            rotations[3].iter().map(|pos| square_at((pos[0], pos[1]))).collect::<Vec<_>>().concat(),
        ];

        let colors = vec![color; vertices[0].len()];

        let mut max = [0, 0];
        let mut min = [WORLD_WIDTH, WORLD_HEIGHT];

        for pos in &rotations[0] {
            if pos[0] > max[0] { max[0] = pos[0]; }
            if pos[1] > max[1] { max[1] = pos[1]; }
            if pos[0] < min[0] { min[0] = pos[0]; }
            if pos[1] < min[1] { min[1] = pos[1]; }
        }

        let width = max[0] - min[0] + 1;
        let height = max[1] - min[1] + 1;

        unsafe {
            gl::GenVertexArrays(1, &mut vao);
            gl::GenBuffers(1, &mut vert_vbo);
            gl::GenBuffers(1, &mut colors_vbo);

            gl::BindVertexArray(vao);

            gl::BindBuffer(gl::ARRAY_BUFFER, vert_vbo);
            gl::BufferData(gl::ARRAY_BUFFER, (vertices[0].len() * mem::size_of::<[u32; 2]>()) as isize, vertices[0].as_ptr().cast(), gl::STATIC_DRAW);
            gl::VertexAttribIPointer(0, 2, gl::UNSIGNED_INT, 0, ptr::null_mut::<i32>().cast());
            gl::EnableVertexAttribArray(0);

            gl::BindBuffer(gl::ARRAY_BUFFER, colors_vbo);
            gl::BufferData(gl::ARRAY_BUFFER, (colors.len() * mem::size_of::<[f32; 4]>()) as isize, colors.as_ptr().cast(), gl::STATIC_DRAW);
            gl::VertexAttribPointer(1, 4, gl::FLOAT, gl::FALSE, 0, ptr::null_mut::<i32>().cast());
            gl::EnableVertexAttribArray(1);
        }


        Piece {
            shader_prog, vao, vert_vbo, colors_vbo, rotations, vertices, pos_changes, curr_rotation: 0, pos: [x_init, WORLD_HEIGHT - height], width, height, color
        }
    }

    pub fn try_move_down(&mut self, base: &Base) -> bool {
        if self.pos[1] == 0 {
            return false;
        }

        self.pos[1] -= 1;
        if base.collides(&self.rotations[self.curr_rotation], self.pos) {
            self.pos[1] += 1;
            return false;
        }

        true
    }

    pub fn try_move_left(&mut self, base: &Base) -> bool {
        if self.pos[0] == 0 {
            return false;
        }

        self.pos[0] -= 1;
        if base.collides(&self.rotations[self.curr_rotation], self.pos) {
            self.pos[0] += 1;
            return false;
        }

        true
    }

    pub fn try_move_right(&mut self, base: &Base) -> bool {
        if self.pos[0] == WORLD_WIDTH - self.width {
            return false;
        }

        self.pos[0] += 1;
        if base.collides(&self.rotations[self.curr_rotation], self.pos) {
            self.pos[0] -= 1;
            return false;
        }

        true
    }

    pub fn try_rotate(&mut self, base: &Base, dir: RotateDir) -> bool {
        let new_rotation;
        let mut new_pos_signed: [i32; 2] = [self.pos[0] as i32, self.pos[1] as i32];

        match dir {
            RotateDir::Left => {
                new_pos_signed[0] += self.pos_changes[self.curr_rotation][0];
                new_pos_signed[1] += self.pos_changes[self.curr_rotation][1];
                new_rotation = if self.curr_rotation == self.rotations.len() - 1 { 0 } else { self.curr_rotation + 1 };
            },
            RotateDir::Right => {
                new_rotation = if self.curr_rotation == 0 { self.rotations.len() - 1 } else { self.curr_rotation - 1 };
                new_pos_signed[0] -= self.pos_changes[new_rotation][0];
                new_pos_signed[1] -= self.pos_changes[new_rotation][1];
            }
        }

        let new_pos = [new_pos_signed[0] as u32, new_pos_signed[1] as u32];

        // check if rotated shape fits (notice that new width is previous height and vice versa)
        if new_pos_signed[0] < 0 || new_pos[0] > WORLD_WIDTH - self.height || new_pos_signed[1] < 0 || new_pos[1] > WORLD_HEIGHT - self.width {
            return false;
        }

        if base.collides(&self.rotations[new_rotation], new_pos) {
            return false;
        }

        self.curr_rotation = new_rotation;
        self.pos = new_pos;

        let tmp = self.width;
        self.width = self.height;
        self.height = tmp;

        unsafe {
            gl::BindBuffer(gl::ARRAY_BUFFER, self.vert_vbo);
            gl::BufferSubData(gl::ARRAY_BUFFER, 0, (self.vertices[0].len() * mem::size_of::<[u32; 2]>()) as isize, self.vertices[self.curr_rotation].as_ptr().cast());
        }

        true
    }

    pub fn add_to(&self, base: &mut Base) {
        base.add(&self.rotations[self.curr_rotation], self.pos, self.color);
    }

    pub fn reset(&mut self) {
        if self.curr_rotation % 2 != 0 {
            let tmp = self.width;
            self.width = self.height;
            self.height = tmp;
        }
        self.curr_rotation = 0;
        self.pos[0] = 0;
        self.pos[1] = WORLD_HEIGHT - self.height;

        unsafe {
            gl::BindBuffer(gl::ARRAY_BUFFER, self.vert_vbo);
            gl::BufferSubData(gl::ARRAY_BUFFER, 0, (self.vertices[0].len() * mem::size_of::<[u32; 2]>()) as isize, self.vertices[self.curr_rotation].as_ptr().cast());
        }
    }

    pub fn draw(&self) {
        unsafe {
            gl::UseProgram(self.shader_prog);

            gl::UniformMatrix4fv(0, 1, gl::FALSE, ORTHO_PROJ.as_ptr());
            gl::Uniform2uiv(1, 1, self.pos.as_ptr());

            gl::BindVertexArray(self.vao);
            gl::DrawArrays(gl::TRIANGLES, 0, self.vertices[0].len() as i32);
        }
    }
}

struct Base {
    shader_prog: u32,
    vao: u32,
    vert_vbo: u32,
    colors_vbo: u32,
    vertices: Vec<[[u32; 2]; 6]>,
    colors: Vec<[[f32; 4]; 6]>,
    grid: Vec<Vec<bool>>
}

impl Base {
    pub fn new(shader_prog: u32) -> Self {
        let grid = vec![vec![false; WORLD_HEIGHT as usize]; WORLD_WIDTH as usize];
        let mut vao = 0;
        let mut vert_vbo = 0;
        let mut colors_vbo = 0;

        unsafe {
            gl::GenVertexArrays(1, &mut vao);
            gl::GenBuffers(1, &mut vert_vbo);
            gl::GenBuffers(1, &mut colors_vbo);

            gl::BindVertexArray(vao);

            gl::BindBuffer(gl::ARRAY_BUFFER, vert_vbo);
            gl::BufferData(gl::ARRAY_BUFFER, (WORLD_WIDTH * WORLD_HEIGHT * mem::size_of::<[[u32; 2]; 6]>() as u32) as isize, ptr::null_mut::<i32>().cast(), gl::DYNAMIC_DRAW);
            gl::VertexAttribIPointer(0, 2, gl::UNSIGNED_INT, 0, ptr::null_mut::<i32>().cast());
            gl::EnableVertexAttribArray(0);

            gl::BindBuffer(gl::ARRAY_BUFFER, colors_vbo);
            gl::BufferData(gl::ARRAY_BUFFER, (WORLD_WIDTH * WORLD_HEIGHT * mem::size_of::<[[f32; 4]; 6]>() as u32) as isize, ptr::null_mut::<i32>().cast(), gl::DYNAMIC_DRAW);
            gl::VertexAttribPointer(1, 4, gl::FLOAT, gl::FALSE, 0, ptr::null_mut::<i32>().cast());
            gl::EnableVertexAttribArray(1);
        }

        Base {
            shader_prog,
            vao,
            vert_vbo, colors_vbo,
            vertices: Vec::new(),
            colors: Vec::new(),
            grid
        }
    }

    pub fn collides(&self, base_shape: &[[u32; 2]], offset: [u32; 2]) -> bool {
        for base_pos in base_shape.iter() {
            let mut pos = base_pos.clone();
            pos[0] += offset[0];
            pos[1] += offset[1];

            assert_eq!(pos[0] < WORLD_WIDTH && pos[1] < WORLD_HEIGHT, true);
            if self.grid[pos[0] as usize][pos[1] as usize] {
                return true;
            }
        }

        false
    }

    pub fn remove_full_lines(&mut self) -> u32 {
        let mut full_lines: Vec<u32> = Vec::new();

        // find full lines
        for y in 0..WORLD_HEIGHT {
            let mut is_full = true;

            for x in 0..WORLD_WIDTH {
                if !self.grid[x as usize][y as usize] {
                    is_full = false;
                    break;
                }
            }

            if is_full {
                full_lines.push(y);
            }
        }

        if full_lines.len() == 0 {
            return 0;
        }

        let mut line_shift = [0; WORLD_HEIGHT as usize];

        // remove full lines in grid and calculate shifting offsets
        for y in &full_lines {
            for i in y+1..WORLD_HEIGHT {
                line_shift[i as usize] += 1;
            }

            for x in 0..WORLD_WIDTH {
                self.grid[x as usize][*y as usize] = false;
            }
        }

        // shift grid lines
        for y in 1..WORLD_HEIGHT {
            for x in 0..WORLD_WIDTH {
                if line_shift[y as usize] > 0 {
                    self.grid[x as usize][(y - line_shift[y as usize]) as usize] = self.grid[x as usize][y as usize];
                    self.grid[x as usize][y as usize] = false;
                }
            }
        }

        // remove full lines in vertices
        for y in &full_lines {
            for i in (0..self.vertices.len()).rev() {
                let square_y = self.vertices[i][1][1];    // y-coord of vertex in lower-left corner
                if square_y == *y {
                    // i hope that swap_remove is deterministic... otherwise, vertices and colors
                    // might become inconsistent!
                    self.vertices.swap_remove(i);
                    self.colors.swap_remove(i);
                } else if square_y > *y {
                    for vertex in &mut self.vertices[i] {
                        vertex[1] -= 1;
                    }
                }
            }
        }

        // do updates and send new vertices to GPU
        unsafe {
            gl::BindBuffer(gl::ARRAY_BUFFER, self.vert_vbo);
            gl::BufferSubData(gl::ARRAY_BUFFER, 0, (self.vertices.len() * mem::size_of::<[[u32; 2]; 6]>()) as isize, self.vertices.as_ptr().cast());
            gl::BindBuffer(gl::ARRAY_BUFFER, self.colors_vbo);
            gl::BufferSubData(gl::ARRAY_BUFFER, 0, (self.colors.len() * mem::size_of::<[[f32; 4]; 6]>()) as isize, self.colors.as_ptr().cast());
        }

        full_lines.len() as u32
    }

    pub fn add(&mut self, base_shape: &[[u32; 2]], offset: [u32; 2], color: [f32; 4]) {
        assert_eq!(self.collides(base_shape, offset), false);

        let mut positions = Vec::new();

        for base_pos in base_shape.iter() {
            let mut pos = base_pos.clone();
            pos[0] += offset[0];
            pos[1] += offset[1];

            assert_eq!(pos[0] < WORLD_WIDTH && pos[1] < WORLD_HEIGHT, true);

            self.grid[pos[0] as usize][pos[1] as usize] = true;
            positions.push(pos);
        }

        let new_vertices = positions.iter().map(|pos| square_at((pos[0], pos[1]))).collect::<Vec<_>>();
        let new_colors = vec![[color; 6]; new_vertices.len()];

        unsafe {
            gl::BindBuffer(gl::ARRAY_BUFFER, self.vert_vbo);
            gl::BufferSubData(gl::ARRAY_BUFFER, (self.vertices.len() * mem::size_of::<[[u32; 2]; 6]>()) as isize, (new_vertices.len() * mem::size_of::<[[u32; 2]; 6]>()) as isize, new_vertices.as_ptr().cast());
            gl::BindBuffer(gl::ARRAY_BUFFER, self.colors_vbo);
            gl::BufferSubData(gl::ARRAY_BUFFER, (self.colors.len() * mem::size_of::<[[f32; 4]; 6]>()) as isize, (new_colors.len() * mem::size_of::<[[f32; 4]; 6]>()) as isize, new_colors.as_ptr().cast());
        }

        self.vertices.extend(new_vertices);
        self.colors.extend(new_colors);
    }

    pub fn draw(&self) {
        let zero_offset = [0, 0];

        unsafe {
            gl::UseProgram(self.shader_prog);

            gl::UniformMatrix4fv(0, 1, gl::FALSE, ORTHO_PROJ.as_ptr());
            gl::Uniform2uiv(1, 1, zero_offset.as_ptr());

            gl::BindVertexArray(self.vao);
            gl::DrawArrays(gl::TRIANGLES, 0, self.vertices.len() as i32 * 6);
        }
    }
} 

struct Tetris<'a> {
    shader_prog: u32,
    pieces: [Piece<'a>; 5],
    active_piece: usize,
    base: Base,
    rng: rand::rngs::ThreadRng,
}

impl<'a> Tetris<'a> {
    pub fn new(shader_prog: u32) -> Self {
        let base: Base = Base::new(shader_prog);
        let mut rng = rand::thread_rng();

        let pieces: [Piece; 5] = [
            // pos_changes[i] gives the relative change in position from the ith to the (i+1)th shape
            Piece::new(shader_prog, 0, &STRAIGHT_SHAPE, [[2, -1], [-2, 2], [1, -2], [-1, 1]], COLOR_BLUE),
            Piece::new(shader_prog, 0, &SQUARE_SHAPE, [[0, 0], [0, 0], [0, 0], [0, 0]], COLOR_YELLOW),
            Piece::new(shader_prog, 0, &T_SHAPE, [[1, 0], [-1, 1], [0, -1], [0, 0]], COLOR_PURPLE),
            Piece::new(shader_prog, 0, &L_SHAPE, [[-1, 1], [0, -1], [0, 0], [1, 0]], COLOR_ORANGE),
            Piece::new(shader_prog, 0, &S_SHAPE, [[1, 0], [-1, 0], [1, 0], [-1, 0]], COLOR_GREEN)
        ];

        Tetris {
            shader_prog,
            pieces,
            active_piece: rng.gen_range(0..5),
            base,
            rng,
        }
    }

    pub fn move_left(&mut self) {
        self.pieces[self.active_piece].try_move_left(&self.base);
    }

    pub fn move_right(&mut self) {
        self.pieces[self.active_piece].try_move_right(&self.base);
    }

    pub fn try_move_down(&mut self) -> bool {
        self.pieces[self.active_piece].try_move_down(&self.base)
    }

    pub fn move_to_bottom(&mut self) {
        while self.pieces[self.active_piece].try_move_down(&self.base) {}
    }

    pub fn new_piece(&mut self) {
        self.pieces[self.active_piece].add_to(&mut self.base);
        self.pieces[self.active_piece].reset();
        self.active_piece = self.rng.gen_range(0..self.pieces.len());
    }

    pub fn remove_full_lines(&mut self) -> u32 {
        self.base.remove_full_lines()
    }

    pub fn rotate_left(&mut self) {
        self.pieces[self.active_piece].try_rotate(&self.base, RotateDir::Left);
    }

    pub fn rotate_right(&mut self) {
        self.pieces[self.active_piece].try_rotate(&self.base, RotateDir::Right);
    }

    pub fn finished(&self) -> bool {
        false
    }

    pub fn draw(&self) {
        self.pieces[self.active_piece].draw();
        self.base.draw();
    }
}

fn main() {
    let mut score = 0;

    let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();

    let (mut win, events) = glfw.create_window(WIN_WIDTH, WIN_HEIGHT, "Tetris - 0",
        glfw::WindowMode::Windowed).expect("Failed to create GLFW window.");

    win.set_key_polling(true);
    win.make_current();

    gl::load_with(|s| glfw.get_proc_address_raw(s));

    let (fb_width, fb_height) = win.get_framebuffer_size();
    let shader_prog;

    unsafe {
        let vertex_shader = gl::CreateShader(gl::VERTEX_SHADER);
        let frag_shader = gl::CreateShader(gl::FRAGMENT_SHADER);
        gl::ShaderSource(vertex_shader, 1, &VERT_SHADER.as_bytes().as_ptr().cast(), &VERT_SHADER.len().try_into().unwrap());
        gl::ShaderSource(frag_shader, 1, &FRAG_SHADER.as_bytes().as_ptr().cast(), &FRAG_SHADER.len().try_into().unwrap());
        gl::CompileShader(vertex_shader);
        gl::CompileShader(frag_shader);

        let mut vertex_shader_res = 0;
        let mut frag_shader_res = 0;
        gl::GetShaderiv(vertex_shader, gl::COMPILE_STATUS, &mut vertex_shader_res);
        gl::GetShaderiv(frag_shader, gl::COMPILE_STATUS, &mut frag_shader_res);
        if vertex_shader_res == 0 || frag_shader_res == 0 {
            if vertex_shader_res == 0 {
                println!("Error in vertex shader.");
            }
            if frag_shader_res == 0 {
                println!("Error in fragment shader.");
            }
            process::exit(1);
        }

        let mut shader_prog_res = 0;
        shader_prog = gl::CreateProgram(); gl::AttachShader(shader_prog, vertex_shader);
        gl::AttachShader(shader_prog, frag_shader);
        gl::LinkProgram(shader_prog);
        gl::GetProgramiv(shader_prog, gl::LINK_STATUS, &mut shader_prog_res);
        if shader_prog_res == 0 {
            let mut info: Vec<u8> = vec![0; 512];
            gl::GetProgramInfoLog(shader_prog, info.len().try_into().unwrap(), ptr::null_mut::<i32>(), info.as_mut_ptr() as *mut i8);
            println!("Error during shader linking.\n{}", String::from_utf8(info).unwrap());
            process::exit(1);
        }
        gl::PolygonMode(gl::FRONT_AND_BACK, gl::FILL);
    }

    let mut t: Tetris = Tetris::new(shader_prog);
    let mut last_down_instant: Instant = Instant::now();
    let mut last_move_instant: Option<Instant> = None;
    let mut last_move: Option<Key> = None;

    while !win.should_close() && !t.finished() {
        let mut key_action: Option<(Key, Action)> = None;

        // handle events
        glfw.poll_events();
        for (_, event) in glfw::flush_messages(&events) {
            key_action = key_action.or(handle_event(&mut win, event));
        }

        // handle activation and disactivation of movement
        if let Some((key, action)) = key_action {
            if action == Action::Press {
                if key == Key::S {
                    t.move_to_bottom();
                    t.new_piece();
                    score += 100 * t.remove_full_lines();
                    win.set_title(format!("Tetris - {}", score).as_str());
                    last_down_instant = Instant::now();
                } else if key == Key::A || key == Key::D {
                    // movement directly upon key press
                    match key {
                       Key::A => t.move_left(),
                       Key::D => t.move_right(),
                       _ => {}
                    }

                    last_move = Some(key);
                    last_move_instant = Some(Instant::now());
                } else if key == Key::Left {
                    t.rotate_left();
                } else if key == Key::Right {
                    t.rotate_right();
                }
            }

            if last_move.is_some() && key == last_move.unwrap() && action == Action::Release {
                last_move_instant = None;
                last_move = None;
            }
        }

        if last_move_instant.is_some() && Instant::now().duration_since(last_move_instant.unwrap()) >= MOVE_TIMEOUT {
            // movement upon key hold
            match last_move {
               Some(Key::A) => t.move_left(),
               Some(Key::D) => t.move_right(),
               _ => {}
            }

            last_move_instant = Some(Instant::now());
        }

        if Instant::now().duration_since(last_down_instant) >= DESCENT_TIMEOUT {
            if !t.try_move_down() {
                t.new_piece();
                score += 100 * t.remove_full_lines();
                win.set_title(format!("Tetris - {}", score).as_str());
            }
            last_down_instant = Instant::now();
        }

        unsafe {
            gl::Clear(gl::COLOR_BUFFER_BIT);
        }
        t.draw();
        win.swap_buffers();

    }

    println!("Final score: {}", score);
} 
