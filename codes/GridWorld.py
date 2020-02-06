from .Q import Q
from .V import V

from typing import Union, Type, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np



def dec2bin(d,nb=0):
    """dec2bin(d,nb=0): conversion nombre entier positif ou nul 
    -> chaîne binaire (si nb>0, complète à gauche par des zéros)"""
    if d==0:
        b="0"
    else:
        b=""
        while d!=0:
            b="01"[d&1]+b
            d=d>>1
    return b.zfill(nb)

def ispuissance(cell,puissance):
    """
    Check if cell contains a power puissance in it
    
    args :
        cell(int) : the value of the cell to test
        puissance(int) : the power
        
    """
    res = dec2bin(cell) 
    res = '0'*(11-len(res)) + res
    return int(res[-(puissance + 1)]) == 1
    
def full_grid_generator(long, large):
    my_grid = np.full((long, large), 15)
    return(my_grid)




class  GridWorld(object):
    """A class to generate and interact with a gridworld environement

    Attributes:
        Q (Q): The empirical Q function of the environement.
        V (V): The empirical Value function of the environement.
        x, y (int): Coordinates of the agent in the grid.
        t (int): The time stamp in the current run.
        path (list): The path taken during the current run

    Methods:
        reset(self, random_init: bool=False, state: Optional[Tuple[int]]=None)
              -> None:
            Resets the agent to the start/random/required position and time to 0.
        step(self, action: Union[str, int]) 
             -> Tuple[Union[Tuple[int], int, bool, None]]:
            Perfomrs the asked action
        dedoublement(self)
            -> None
            Doubles the Grid in order to model two-way roads from one-way road 
            It updates the start_coordinates, the end_coordinates and the grid to model
            a two-way road and adds the start positions of the taxis in the grid.
        taxis_trips(self)
            -> None
            Create the path for every taxis on the Grid.
            For each taxis the attribute traxi.trip is created.It describes the road-way
            the taxi will hit.
        next_table(self)
            -> None
            Update the table from t to t+1
            Every taxis move on his next position
        render_path_and_V(self, scale: int=1.5) -> None:
            Plots the path taken by the agent and the Value function.
        render_board(self, scale= float:1.5, show: bool=True,
                     fig: bool=False) -> None:
            displays a figure with the labirynth.
        render_path(self, fig: bool=False, show: bool=True) -> None:
            Plots the path taken by the agent up to now
        render_V(self, fig: bool=False, show: bool=True,from: str='Q') -> None:
            Plots the Value function

    Constants:
        ACTION_SPACE (list): The possible actions in int
        ACTION_DICT (dict): The possible actions in strings
            "up", "right", "down", "left"
    """
    ACTION_SPACE = [8,4,2,1,0]
    ACTION_DICT = {"stop":0,"up": 8, "right":4, "down": 2, "left": 1}
    _TRACE = [[[0,1], [0,0]], [[1,1], [0,1]], [[0,1], [1,1]], [[0,0], [0,1]]]

    
    def __init__(self, full_grid ,taxis, init_position,end_position) -> None:
        """Initializes the grid

        If a filename is passed then the grid is parsed, otherwise, if a tuple
        is provided, then a random grid is generated.
        Also initialises a Q and a Value function for the grid

        Args:
            grid (str, tuple): A file name or the size of the requested grid.
                The grid is defined with the following synthax:
                  00000000000 (0) blank cell
                  00000001000 (8) wall up
                  00000000100 (4) wall right
                  00000000010 (2) wall down 
                  00000000001 (1) wall left 
                  00000010000 (16) start
                  00000100000 (32) end
                  00001000000 (64) taxi in the cell

                Any combination of those basics cells are possible
                Hence 100101=37 is an end cell with a wall to the right and 
                a wall to the left

        Returns:
            None
        """
        self._IA0 = init_position
        self._grid = full_grid
        self.taxis = taxis
        #Création d'une matrice à partir d'un fichier
        self._start_coordinates = init_position
        self._end_coordinates = end_position
        
        
        for trip in self.taxis :
            departure = trip.departure
            arrival = trip.arrival
            
            for cell in range(len(self._grid)):
                if ispuissance(self._grid[cell, departure[1]],3):
                    self._grid[cell, departure[1]] -= 8
                    
                if ispuissance(self._grid[cell, departure[1]],1):
                    self._grid[cell, departure[1]] -= 2
            
            for cell in range(len(self._grid[0])):
                if ispuissance(self._grid[departure[0], cell],2):
                    self._grid[departure[0], cell] -= 4 
                if ispuissance(self._grid[departure[0], cell],0):
                    self._grid[departure[0], cell] -= 1    
            
            for cell in range(len(self._grid)):
                if ispuissance(self._grid[cell, arrival[1]],3):
                    self._grid[cell,arrival[1]] -= 8
                if ispuissance(self._grid[cell, arrival[1]],1):
                    self._grid[cell, arrival[1]] -= 2
            
            for cell in range(len(self._grid[0])):
                if ispuissance(self._grid[arrival[0], cell],2):
                    self._grid[arrival[0], cell] -= 4
                if ispuissance(self._grid[arrival[0], cell],0):
                    self._grid[arrival[0], cell] -= 1


        
        for cell in range(len(self._grid[0])) :            
            if not(ispuissance(self._grid[0][cell],3)):
                    self._grid[0][cell] += 8    
        for cell in range(len(self._grid)) :           
            if not(ispuissance(self._grid[cell][len(self._grid[0]) -1],2)):
                    self._grid[cell][len(self._grid[0]) -1] += 4         
        for cell in range(len(self._grid)) :
            if not(ispuissance(self._grid[cell][0],0)):
                    self._grid[cell][0] += 1             
        for cell in range(len(self._grid[0])) :
            if not(ispuissance(self._grid[len(self._grid) - 1][cell], 1)) :
                    self._grid[len(self._grid) - 1][cell] += 2   
        
        self.dedoublement()
        self.taxis_trips()

        self.Q = Q(list(self.ACTION_DICT.keys()), state_shape=self._grid.shape)
        self.V = V(state_shape=self._grid.shape)
        self.reset()
        
        return None

    def dedoublement (self):
        
        my_new_grid = np.zeros((len(self._grid) * 2, len(self._grid[0]) * 2),dtype='int32')
        self._start_coordinate = (self._start_coordinates[0] * 2, self._start_coordinates[1] * 2)
        self._end_coordinates = (self._end_coordinates[0] * 2 + 1, self._end_coordinates[1] * 2 + 1)
        my_new_grid[self._start_coordinates[0], self._start_coordinates[1]] += 16
        my_new_grid[self._end_coordinates[0], self._end_coordinates[1]] += 32 
        
        
        
        
        for i in range(len(self._grid)) :
            for j in range(len(self._grid[0])) :
                if ispuissance(self._grid[i][j], 0) :
                    my_new_grid[i * 2][j * 2] += 1
                    my_new_grid[i * 2 + 1, j * 2] += 1
                if ispuissance(self._grid[i, j], 1) :
                    my_new_grid[i * 2 + 1][j * 2] += 2
                    my_new_grid[i * 2 + 1][j * 2 + 1] += 2
                if ispuissance(self._grid[i][j], 2) :
                    my_new_grid[i * 2 + 1][j * 2 + 1] += 4
                    my_new_grid[i * 2][j * 2 + 1] += 4
                if ispuissance(self._grid[i, j],3) :
                    my_new_grid[i * 2 , j * 2] += 8
                    my_new_grid[i * 2, j * 2 +1] += 8
                    
                
                
        self._grid = my_new_grid

        for taxi in self.taxis :
            if taxi.departure[0] == taxi.arrival[0] :
                if taxi.departure[1] > taxi.arrival[1] :
                    taxi.departure = [taxi.departure[0] * 2, taxi.departure[1] * 2]
                    taxi.arrival = [taxi.arrival[0] * 2, taxi.arrival[1] * 2]
                        
                else :
                    taxi.departure = [taxi.departure[0] * 2 + 1, taxi.departure[1] * 2 + 1]
                    taxi.arrival = [taxi.arrival[0] * 2 + 1, taxi.arrival[1] * 2 + 1]
                    
            elif taxi.departure[1] == taxi.arrival[1] :
                    if taxi.departure[0] > taxi.arrival[0] :
                        taxi.departure = [taxi.departure[0] * 2 + 1, taxi.departure[1] * 2 + 1]
                        taxi.arrival = [taxi.arrival[0] * 2 + 1, taxi.arrival[1] * 2 + 1]
                        
                    else :
                        taxi.departure = [taxi.departure[0] * 2, taxi.departure[1] * 2]
                        taxi.arrival = [taxi.arrival[0] * 2, taxi.arrival[1] * 2]

            else :
                if taxi.departure[0] < taxi.arrival[0] :
                    if taxi.arrival[1] > taxi.departure[1] :
                        taxi.departure = [taxi.departure[0] * 2 + 1, taxi.departure[1] * 2 + 1]
                    else :
                        taxi.departure = [taxi.departure[0] * 2, taxi.departure[1] * 2]
                    taxi.arrival = [taxi.arrival[0] * 2, taxi.arrival[1] * 2]
                    
                else :
                    
                    taxi.departure = [taxi.departure[0] * 2 + 1, taxi.departure[1] * 2 + 1]
                    if taxi.arrival[1] > taxi.departure[1] :
                        taxi.arrival = [taxi.arrival[0] * 2 + 1, taxi.arrival[1] * 2 + 1]
                    taxi.arrival = [taxi.arrival[0] * 2, taxi.arrival[1] * 2]
            
            
            taxi.position = taxi.departure
            self._grid[taxi.position[0], taxi.position[1]] += 64
            
        return None

    def taxis_trips(self):

        for taxi in self.taxis :
            taxi.trip = []
            if taxi.departure[0] == taxi.arrival[0] :
                distance = abs(taxi.departure[1] - taxi.arrival[1])
                if taxi.departure[1] > taxi.arrival[1] :
                    taxi.trip = [(taxi.departure[0],taxi.departure[1] - i) for i in range(0,distance+1)]    
                else :
                    taxi.trip = [(taxi.departure[0],taxi.departure[1] + i) for i in range(0,distance+1)]                    

            elif taxi.departure[1] == taxi.arrival[1] :
                distance = abs(taxi.departure[0] - taxi.arrival[0])
                if taxi.departure[0] > taxi.arrival[0] :
                    taxi.trip = [(taxi.departure[0] - i,taxi.departure[1] ) for i in range(0,distance+1)]   
                        
                else :
                    taxi.trip = [(taxi.departure[0] + i,taxi.departure[1]) for i in range(0,distance+1)]   

            else :
                distancex = abs(taxi.departure[0] - taxi.arrival[0])
                distancey = abs(taxi.departure[1] - taxi.arrival[1])
                if taxi.departure[0] < taxi.arrival[0] :
                    if taxi.arrival[1] > taxi.departure[1] :
                        taxi.trip = [(taxi.departure[0],taxi.departure[1] + i) for i in range(0,distancey+1)] +\
                        [(taxi.departure[0] + i ,taxi.arrival[1] ) for i in range(1,distancex+1)]
                    else :
                        taxi.trip = [(taxi.departure[0],taxi.departure[1] - i) for i in range(0,distancey+1)] +\
                        [(taxi.departure[0] + i ,taxi.arrival[1] ) for i in range(1,distancex+1)]
                    
                else :
                    if taxi.arrival[1] > taxi.departure[1] :
                        taxi.trip = [(taxi.departure[0] - i ,taxi.departure[1] ) for i in range(0,distancex+1)] +\
                        [(taxi.arrival[0],taxi.departure[1] + i) for i in range(1,distancey+1)] 
                    else :    
                        taxi.trip = [(taxi.departure[0] - i ,taxi.departure[1] ) for i in range(0,distancex+1)] +\
                        [(taxi.arrival[0],taxi.departure[1] - i) for i in range(1,distancey+1)] 
                        
        return None
            
     
    def next_table(self):
        
        for taxi in self.taxis:
            self._grid[taxi.position[0], taxi.position[1]] -= 64
            duration_trip = len(taxi.trip)
            taxi.position = taxi.trip[(self.t)%duration_trip]
            self._grid[taxi.position[0], taxi.position[1]] += 64
            
        self.Q = Q(list(self.ACTION_DICT.keys()), state_shape=self._grid.shape)
        self.V = V(state_shape=self._grid.shape)
        
        return None
        
    def reset(self,t = 0) -> Tuple[int]:
        """Resets the agent to the start/random/required position and time to 0.

        Args:
            t(int) : the time the map should be reseted.

        Returns:
            state (tuple): posiiton of the agent
        """
        if t == 0 :
            self._IA = self._IA0

        self._current_cell = self._grid[self._IA[0], self._IA[1]]

        self.t = t
        self.path = []
        
        for taxi in self.taxis :
            trip_duration = len(taxi.trip)
            taxi.position = taxi.trip[t%trip_duration]

        return self._IA

    def step(self, action: Union[str, int])\
            -> Tuple[Union[Tuple[int], int, bool, None]]:
        """Perfomrs the asked action

        Checks if the action is possible, and perfoms it, the agent stays in
        place. Also checks the reward. The reward function is: 1 if done, 0
        otherwise.

        Args:
            action (str, int): The action to perform. 

        Return:
            state (tuple): The new position of the agent.
            reward (int): The reward of the action
            done (True): If True, the episode is over.
            None
        """
        if isinstance(action, str):
            action = self.ACTION_DICT[action]

        if not self._check_action_(action):
            pass

        else:
            x_moove, y_moove = self.action_results_(action)

            self._IA = (self._IA[0] + x_moove,self._IA[1] + y_moove) 
            
            
            self.path.append([self._IA[0],self._IA[1]])

        self.t += 1
        self._current_cell = self._grid[self._IA[0],self._IA[1]]
        done = self._check_terminate_()
        if done:
            reward = 1 
        else :
            reward = 0
#Fonction reward à programmer  plus tard et redefinir le done
        return self._IA, reward, done, None
        
    def render_path_and_V(self, scale: int=1.5) -> None:
        """Plots the path taken by the agent and the Value function.

        Args:
            scale (float): size of the figures
        """
        shape = tuple([_*0.5 for _ in self._grid.shape])
        plt.figure(figsize=(shape[1]*scale*2+1, shape[0]*scale))
        plt.subplot(1,2,1)
        self.render_path(fig=True, show=False)
        plt.subplot(1,2,2)
        self.render_V(fig=True, show=False)
        plt.show()

        return None

    def render_board(self, scale: float=1.5, show: bool=True,
                     fig: bool=False, save = None) -> None:
        """displays a figure with the labirynth.

        Args:
            scale (float): size of the figures
            show (bool): Wether to show the figure
            fig (bool): If False, plots on the currently opened figure.

        return:
            None
        """
        if not fig:
            shape = tuple([_*0.5 for _ in self._grid.shape])
            plt.figure(figsize=(shape[1]*scale, shape[0]*scale))
        
        plt.ylim(self._grid.shape[0]+0.1, -0.1)
        plt.xlim(-0.1,self._grid.shape[1]+0.1)

        for i in range(self._grid.shape[0]):
            for j in range(self._grid.shape[1]):
                for ind,_ in enumerate(bin(256+self._grid[i,j])[-4:]):
                    if _ == "1": 
                        x_modif, y_modif = self._TRACE[ind]
                        x = [j+x_modif[0], j+x_modif[1]]
                        y = [i+y_modif[0], i+y_modif[1]]
                        plt.plot(x,y,"k-", linewidth=5)
             
        plt.text(x=self._start_coordinates[1]+0.5,
                 y=self._start_coordinates[0]+0.5, s="START",
                    bbox={'facecolor':'purple','alpha':1,'edgecolor':'none','pad':1},
                    ha='center', va='center', color='white')
        plt.text(x=self._end_coordinates[1]+0.5,
                 y=self._end_coordinates[0]+0.5, s="END",
                    bbox={'facecolor':'purple','alpha':1,'edgecolor':'none','pad':1},
                    ha='center', va='center', color='white')

        for taxi in self.taxis :
            plt.text(x=taxi.position[1]+0.5,
                 y=taxi.position[0]+0.5,s = "taxi",
                    bbox={'facecolor':'yellow','alpha':1,'edgecolor':'none','pad':1},
                    ha='center', va='center', color='black')

            plt.text(x=self._IA[1]+0.5,
                 y=self._IA[0]+0.5,s = " MG ",
                    bbox={'facecolor':'blue','alpha':1,'edgecolor':'none','pad':1},
                    ha='center', va='center', color='black')
       
        if show:
            plt.show()
        
        if save != None:
            plt.savefig(save)

        return None

    def render_path(self, fig: bool=False, show: bool=True) -> None:
        """Plots the path taken by the agent up to now

        Args:
            show (bool): Wether to show the figure
            fig (bool): If False, plots on the currently opened figure.

        return:
            None
        """
        self.render_board(show=False, fig=fig)
        current_pos = (self._start_coordinates[0], self._start_coordinates[1])
        for next_pos in self.path:
            y = [current_pos[0]+1/2, next_pos[0]+1/2]
            x = [current_pos[1]+1/2, next_pos[1]+1/2]
            plt.plot(x,y,"b-", linewidth=5)
            current_pos = next_pos

        plt.title("Path taken by the agent, %i steps taken." % len(self.path))

        if show:
            plt.show()

        return None

    def render_V(self, fig: bool=False, show: bool=True, f: str='Q') -> None:
        """Plots the Value function

        Args:
            show (bool): Wether to show the figure
            fig (bool): If False, plots on the currently opened figure.
            f (str): from which source to plot the Value function

        return:
            None
        """
        self.Q.get_V()
        self.render_board(show=False, fig=fig)
        if f == 'Q':
            plt.pcolormesh(self.Q.V, cmap="hot")

        if f == 'V':
            plt.pcolormesh(self.V.W, cmap="hot")

        ax = plt.gca()
        ax.set_aspect('equal')
        plt.title("State of the value function")

        if show:
            plt.show()

        return None

    def _check_action_(self, action: int) -> bool:
        """Checks if the proposed action is possible to execute

        Args:
            action (int): The action to perform

        Return:
            None
        """
        
        
        if action == 8 :
            if (self._IA[1] % 2) == 1 and  not(ispuissance(self._grid[self._IA[0], self._IA[1]], 3)) :
                if (self._IA[0] + 1) % 2 == 0 :
                    if not(ispuissance(self._grid[self._IA[0] - 1, self._IA[1]], 2)) :
                        return(not(ispuissance(self._grid[self._IA[0] - 1, self._IA[1] + 1],6)))
                    else :
                        return(True)
                else :
                    if not(ispuissance(self._grid[self._IA[0] - 1, self._IA[1]], 0)) :
                        return(not(ispuissance(self._grid[self._IA[0] - 1, self._IA[1] - 1],6)))
                    else :
                        return(True)
            else :
                return(False)
        
        if action == 2 :
            if (self._IA[1] % 2) == 0 and not(ispuissance(self._grid[self._IA[0], self._IA[1]], 1)) :
                if self._IA[0] + 1 % 2 == 0 :
                    if not(ispuissance(self._grid[self._IA[0] + 1, self._IA[1]], 2)) :
                        return(not(ispuissance(self._grid[self._IA[0] + 1, self._IA[1] + 1], 6)))
                    else :
                        return(True)
                else :
                    if not(ispuissance(self._grid[self._IA[0] + 1, self._IA[1]], 0)) :
                        return(not(ispuissance(self._grid[self._IA[0] + 1, self._IA[1] - 1], 6)))
                    else :
                        return(True)
            else :
                return(False)
                
        if action == 4 :
            if self._IA[0] % 2 == 1 and not(ispuissance(self._grid[self._IA[0], self._IA[1]], 2)) :
                if self._IA[1] + 1 % 2 == 0 :
                    if not(ispuissance(self._grid[self._IA[0], self._IA[1] + 1], 3)) :
                        return(not(ispuissance(self._grid[self._IA[0] - 1, self._IA[1] + 1], 6)))
                    else :
                        return(True)
                else:
                    if not(ispuissance(self._grid[self._IA[0], self._IA[1] + 1], 1)) :
                        return(not(ispuissance(self._grid[self._IA[0] + 1, self._IA[1] + 1], 6)))
                    else :
                        return(True)
            else :
                return(False)
        
        if action == 1 :
            if (self._IA[0] % 2 == 0) and not(ispuissance(self._grid[self._IA[0], self._IA[1]], 0)) :
                if self._IA[1] + 1 % 2 == 0 :
                    if not(ispuissance(self._grid[self._IA[0], self._IA[1] - 1], 3)) :
                        return(not(ispuissance(self._grid[self._IA[0] - 1, self._IA[1] - 1], 6)))
                    else :
                        return(True)
                else :
                    if not(ispuissance(self._grid[self._IA[0], self._IA[1]], 1)) :
                        return(not(ispuissance(self._grid[self._IA[0] + 1, self._IA[1] - 1], 6)))
                    else :
                        return(True)
                        
            else :
                return(False)
               
        if action == 0 :
            if self._grid[self._IA[0], self._IA[1]] in [0,3,6,9,12] :
                return(False) #impossible de s arreter dans un coin ou une intersection
            else :
                if not(ispuissance(self._grid[self._IA[0], self._IA[1]], 3) or ispuissance(self._grid[self._IA[0], self._IA[1]], 1)):
                    if self._IA[1] % 2 == 0 :
                        if not(ispuissance(self._grid[self._IA[0], self._IA[1]], 3)) :
                            return(not(ispuissance(self._grid[self._IA[0] - 1, self._IA[1]], 6)))
                        else :
                            return(True)
                    else :
                        if not(ispuissance(self._grid[self._IA[0] - 1, self._IA[1]], 1)) :
                            return(not(ispuissance(self._grid[self._IA[0] + 1, self._IA[1]], 6)))
                        else :
                            return(True)
                else :
                    if self._IA[0] % 2 == 0 :
                        if not(ispuissance(self._grid[self._IA[0] - 1, self._IA[1]], 2)) :
                            return(not(ispuissance(self._grid[self._IA[0], self._IA[1] + 1], 6)))
                        else :
                            return(True)
                    else :
                        if not(ispuissance(self._grid[self._IA[0] - 1, self._IA[1]], 0)) :
                            return(not(ispuissance(self._grid[self._IA[0], self._IA[1] - 1], 6)))
                        else :
                            return(True)
        
    def _check_terminate_(self) -> bool:
        """If the state is terminal returns True, False otherwise

        Returns:
            terminate (bool)
        """
        state = self._current_cell 
        return ispuissance(state,5)

    @staticmethod
    def action_results_(action: int) -> Tuple[int]:
        """ returns the state modification of an action

        Args:
            action (str, int): The action to perform. 

        Returns:
            state modif (tuple)
        """
        if action == 8:
            return -1, 0

        elif action == 4:
            return 0, 1

        elif action == 2:
            return 1, 0

        elif action == 1:
            return 0, -1
        
        elif action == 0:
            return 0, 0


