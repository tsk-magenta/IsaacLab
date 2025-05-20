import asyncio

class Particle():
    def __init__(self):
        print("Particle Started")

        # self._appwindow = omni.appwindow.get_default_app_window()
        # self._input = carb.input.acquire_input_interface()
        # self._keyboard = self._appwindow.get_keyboard()

        # self._keyboard_sub = self._input.subscribe_to_keyboard_events(
        #     self._keyboard,
        #     lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        # )

        # self._is_creating_particles = False
        # self._particle_task = None
        # self._sphere_id_counter = itertools.count()