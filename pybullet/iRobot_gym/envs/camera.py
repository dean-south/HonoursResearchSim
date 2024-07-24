import pybullet as p

class CameraController:
    def __init__(self):
        self.camera_distance = 11
        self.camera_yaw = 0
        self.camera_pitch = -89
        self.camera_target_position = [0, 0, 0]
        
        self.move_speed = 0.25
        self.rotate_speed = 0.5
        self.zoom_speed = 0.5

        self.last_mouse_x = None
        self.last_mouse_y = None
        self.is_shift_pressed = False

    def update(self):
        keys = p.getKeyboardEvents()

        # Check if shift is pressed
        self.is_shift_pressed = p.B3G_SHIFT in keys and keys[p.B3G_SHIFT] & p.KEY_IS_DOWN

        # Handle camera movement (arrow keys)
        if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
            self.camera_target_position[1] += self.move_speed
        if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
            self.camera_target_position[1] -= self.move_speed
        if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
            self.camera_target_position[0] -= self.move_speed
        if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
            self.camera_target_position[0] += self.move_speed

        camera_parameters = p.getDebugVisualizerCamera()

        # Extracting the relevant parameters
        yaw = camera_parameters[8]
        pitch = camera_parameters[9]
        distance = camera_parameters[10]

        # Update camera view matrix
        p.resetDebugVisualizerCamera(
            distance, yaw, pitch, self.camera_target_position)
