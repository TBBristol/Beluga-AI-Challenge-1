import pygame
import json
import typing
import math

class Beluga_board:
    def __init__(self, json_file: str):
        with open(json_file, "r") as f:
            self.data = json.load(f)
        self.trailers_beluga = self.data["trailers_beluga"]
        self.trailers_factory = self.data["trailers_factory"]
        self.hangars = self.data["hangars"]
        self.jig_types = self.data["jig_types"]
        self.racks = self.data["racks"]
        self.jigs = self.data["jigs"]
        self.production_lines = self.data["production_lines"]
        self.flights = self.data["flights"]
        self.window_size_x = 1200
        self.window_size_y = 1000
    
        
        #todo add the rest of the data for abstract uncertainty model

    def _draw_racks(self):
        num_racks = len(self.racks)
        self.rows = math.ceil(num_racks / 10) 
        self.racks_per_row = math.ceil(num_racks / self.rows ) 
        self.rack_draw_width = (self.window_size_x - 200 - ((self.racks_per_row-1)*10)) / self.racks_per_row
        self.rack_draw_height = 200/self.rows
        row = 0
        row_counter = 0
        for i, rack in enumerate(self.racks):
            
            
            pygame.draw.rect(self.screen, (0, 0, 0), (100+ row_counter* self.rack_draw_width + 10*row_counter, #x_start
                                                      self.window_size_y/3 + row * self.rack_draw_height + row*20, #y_start
                                                      self.rack_draw_width, #width
                                                      self.rack_draw_height)) #height
            row_counter += 1
            if row_counter >= self.racks_per_row:
                row += 1
                row_counter = 0

    def draw_rack_jigs(self):
        self.font = pygame.font.SysFont("Arial", 10)
        row = 0
        row_counter = 0
        for i, rack in enumerate(self.racks):
            
            jigs = rack["jigs"]
            empty_space = rack["size"]
            for j, jig in enumerate(jigs):
                num_jigs = len(jigs)
                jig_type = self.jigs[jig]["type"]
                jig_empty = self.jigs[jig]["empty"]
                jig_name = self.jigs[jig]["name"]


                if jig_empty:
                    colour = (0,255,0) #green
                    jig_size = self.jig_types[jig_type]["size_empty"]
                    
                else:
                    colour = (255,0,0) #red
                    jig_size = self.jig_types[jig_type]["size_loaded"]


                empty_space -= jig_size
                text_surface = self.font.render(jig_name+ ' '+f"({jig_size})", False, (colour))
                text_loc = (100+ row_counter* self.rack_draw_width + 10*row_counter +self.rack_draw_width/2-21,
                            (self.window_size_y/3) + self.rack_draw_height/2 + num_jigs*10/2 - 12 - j*10 +row*self.rack_draw_height + 20*row) #co-ords starts top left x, y
                self.screen.blit(text_surface,text_loc)

            
            
            text_surf_rack_size = self.font.render(f"Empty_space: {empty_space}", False, (0,0,0))
            text_loc_rack_size = (100+ row_counter* self.rack_draw_width + 10*row_counter +(self.rack_draw_width/2)-33,
                                   (self.window_size_y/3) -12 + row * self.rack_draw_height +row*20)
            self.screen.blit(text_surf_rack_size,text_loc_rack_size)
            row_counter += 1
            if row_counter >= self.racks_per_row:
                row += 1
                row_counter = 0

    def draw_trailers(self):
        for i, trailer in enumerate(self.trailers_beluga):
            trailer_name = trailer["name"]
           



                


    def _render_frame(self):
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((self.window_size_x, self.window_size_y))  # Adjust size as needed
        pygame.display.set_caption("Beluga Board")
       
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            self.screen.fill((255, 255, 255))  # White background

            self._draw_racks()
            self.draw_rack_jigs()




            pygame.display.flip()
        
        pygame.quit()


if __name__ == "__main__":
    Beluga_board = Beluga_board("/Users/ha24583/Documents/GitHub/Beluga-AI-Challenge-internal/tools/output/problem_j355_r20_oc23_f199_s135_133.json")
    Beluga_board._render_frame()
    print("stop")
    
