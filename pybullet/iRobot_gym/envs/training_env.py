import numpy as np

def set_constrained_env(maze_size=64, wall_rate=0.45, show_walls=True):

    obj_file = ''

    if show_walls:
        for i in range(-maze_size//2, maze_size//2):
            obj_file += f'''
    v {0.05-maze_size//2} {i} 0
    v {0.05-maze_size//2} {i+1} 0
    v {0.05-maze_size//2} {i+1} 0.5
    v {0.05-maze_size//2} {i} 0.5
    v {-0.05-maze_size//2} {i} 0
    v {-0.05-maze_size//2} {i+1} 0
    v {-0.05-maze_size//2} {i+1} 0.5
    v {-0.05-maze_size//2} {i} 0.5

    v {0.05+maze_size//2} {i} 0
    v {0.05+maze_size//2} {i+1} 0
    v {0.05+maze_size//2} {i+1} 0.5
    v {0.05+maze_size//2} {i} 0.5
    v {-0.05+maze_size//2} {i} 0
    v {-0.05+maze_size//2} {i+1} 0
    v {-0.05+maze_size//2} {i+1} 0.5
    v {-0.05+maze_size//2} {i} 0.5

    v {i} {0.05-maze_size//2} 0
    v {i+1} {0.05-maze_size//2} 0
    v {i+1} {0.05-maze_size//2} 0.5
    v {i} {0.05-maze_size//2} 0.5
    v {i} {-0.05-maze_size//2} 0
    v {i+1} {-0.05-maze_size//2} 0
    v {i+1} {-0.05-maze_size//2} 0.5
    v {i} {-0.05-maze_size//2} 0.5

    v {i} {0.05+maze_size//2} 0
    v {i+1} {0.05+maze_size//2} 0
    v {i+1} {0.05+maze_size//2} 0.5
    v {i} {0.05+maze_size//2} 0.5
    v {i} {-0.05+maze_size//2} 0
    v {i+1} {-0.05+maze_size//2} 0
    v {i+1} {-0.05+maze_size//2} 0.5
    v {i} {-0.05+maze_size//2} 0.5

    f {1 + (i + maze_size//2)*32} {2 + (i + maze_size//2)*32} {3 + (i + maze_size//2)*32}
    f {1 + (i + maze_size//2)*32} {3 + (i + maze_size//2)*32} {4 + (i + maze_size//2)*32}
    f {5 + (i + maze_size//2)*32} {6 + (i + maze_size//2)*32} {7 + (i + maze_size//2)*32}
    f {5 + (i + maze_size//2)*32} {7 + (i + maze_size//2)*32} {8 + (i + maze_size//2)*32}
    f {1 + (i + maze_size//2)*32} {5 + (i + maze_size//2)*32} {6 + (i + maze_size//2)*32}
    f {1 + (i + maze_size//2)*32} {6 + (i + maze_size//2)*32} {2 + (i + maze_size//2)*32}
    f {2 + (i + maze_size//2)*32} {6 + (i + maze_size//2)*32} {7 + (i + maze_size//2)*32}
    f {2 + (i + maze_size//2)*32} {7 + (i + maze_size//2)*32} {3 + (i + maze_size//2)*32}
    f {3 + (i + maze_size//2)*32} {7 + (i + maze_size//2)*32} {8 + (i + maze_size//2)*32}
    f {3 + (i + maze_size//2)*32} {8 + (i + maze_size//2)*32} {4 + (i + maze_size//2)*32}
    f {4 + (i + maze_size//2)*32} {8 + (i + maze_size//2)*32} {5 + (i + maze_size//2)*32}
    f {4 + (i + maze_size//2)*32} {5 + (i + maze_size//2)*32} {1 + (i + maze_size//2)*32}

    f {3 + (i + maze_size//2)*32} {2 + (i + maze_size//2)*32} {1 + (i + maze_size//2)*32}
    f {4 + (i + maze_size//2)*32} {3 + (i + maze_size//2)*32} {1 + (i + maze_size//2)*32}
    f {7 + (i + maze_size//2)*32} {6 + (i + maze_size//2)*32} {5 + (i + maze_size//2)*32}
    f {8 + (i + maze_size//2)*32} {7 + (i + maze_size//2)*32} {5 + (i + maze_size//2)*32}
    f {6 + (i + maze_size//2)*32} {5 + (i + maze_size//2)*32} {1 + (i + maze_size//2)*32}
    f {2 + (i + maze_size//2)*32} {6 + (i + maze_size//2)*32} {1 + (i + maze_size//2)*32}
    f {7 + (i + maze_size//2)*32} {6 + (i + maze_size//2)*32} {2 + (i + maze_size//2)*32}
    f {3 + (i + maze_size//2)*32} {7 + (i + maze_size//2)*32} {2 + (i + maze_size//2)*32}
    f {8 + (i + maze_size//2)*32} {7 + (i + maze_size//2)*32} {3 + (i + maze_size//2)*32}
    f {4 + (i + maze_size//2)*32} {8 + (i + maze_size//2)*32} {3 + (i + maze_size//2)*32}
    f {5 + (i + maze_size//2)*32} {8 + (i + maze_size//2)*32} {4 + (i + maze_size//2)*32}
    f {1 + (i + maze_size//2)*32} {5 + (i + maze_size//2)*32} {4 + (i + maze_size//2)*32}

    f {1 + (i + maze_size//2)*32 + 8} {2 + (i + maze_size//2)*32 + 8} {3 + (i + maze_size//2)*32 + 8}
    f {1 + (i + maze_size//2)*32 + 8} {3 + (i + maze_size//2)*32 + 8} {4 + (i + maze_size//2)*32 + 8}
    f {5 + (i + maze_size//2)*32 + 8} {6 + (i + maze_size//2)*32 + 8} {7 + (i + maze_size//2)*32 + 8}
    f {5 + (i + maze_size//2)*32 + 8} {7 + (i + maze_size//2)*32 + 8} {8 + (i + maze_size//2)*32 + 8}
    f {1 + (i + maze_size//2)*32 + 8} {5 + (i + maze_size//2)*32 + 8} {6 + (i + maze_size//2)*32 + 8}
    f {1 + (i + maze_size//2)*32 + 8} {6 + (i + maze_size//2)*32 + 8} {2 + (i + maze_size//2)*32 + 8}
    f {2 + (i + maze_size//2)*32 + 8} {6 + (i + maze_size//2)*32 + 8} {7 + (i + maze_size//2)*32 + 8}
    f {2 + (i + maze_size//2)*32 + 8} {7 + (i + maze_size//2)*32 + 8} {3 + (i + maze_size//2)*32 + 8}
    f {3 + (i + maze_size//2)*32 + 8} {7 + (i + maze_size//2)*32 + 8} {8 + (i + maze_size//2)*32 + 8}
    f {3 + (i + maze_size//2)*32 + 8} {8 + (i + maze_size//2)*32 + 8} {4 + (i + maze_size//2)*32 + 8}
    f {4 + (i + maze_size//2)*32 + 8} {8 + (i + maze_size//2)*32 + 8} {5 + (i + maze_size//2)*32 + 8}
    f {4 + (i + maze_size//2)*32 + 8} {5 + (i + maze_size//2)*32 + 8} {1 + (i + maze_size//2)*32 + 8}

    f {3 + (i + maze_size//2)*32 + 8} {2 + (i + maze_size//2)*32 + 8} {1 + (i + maze_size//2)*32 + 8}
    f {4 + (i + maze_size//2)*32 + 8} {3 + (i + maze_size//2)*32 + 8} {1 + (i + maze_size//2)*32 + 8}
    f {7 + (i + maze_size//2)*32 + 8} {6 + (i + maze_size//2)*32 + 8} {5 + (i + maze_size//2)*32 + 8}
    f {8 + (i + maze_size//2)*32 + 8} {7 + (i + maze_size//2)*32 + 8} {5 + (i + maze_size//2)*32 + 8}
    f {6 + (i + maze_size//2)*32 + 8} {5 + (i + maze_size//2)*32 + 8} {1 + (i + maze_size//2)*32 + 8}
    f {2 + (i + maze_size//2)*32 + 8} {6 + (i + maze_size//2)*32 + 8} {1 + (i + maze_size//2)*32 + 8}
    f {7 + (i + maze_size//2)*32 + 8} {6 + (i + maze_size//2)*32 + 8} {2 + (i + maze_size//2)*32 + 8}
    f {3 + (i + maze_size//2)*32 + 8} {7 + (i + maze_size//2)*32 + 8} {2 + (i + maze_size//2)*32 + 8}
    f {8 + (i + maze_size//2)*32 + 8} {7 + (i + maze_size//2)*32 + 8} {3 + (i + maze_size//2)*32 + 8}
    f {4 + (i + maze_size//2)*32 + 8} {8 + (i + maze_size//2)*32 + 8} {3 + (i + maze_size//2)*32 + 8}
    f {5 + (i + maze_size//2)*32 + 8} {8 + (i + maze_size//2)*32 + 8} {4 + (i + maze_size//2)*32 + 8}
    f {1 + (i + maze_size//2)*32 + 8} {5 + (i + maze_size//2)*32 + 8} {4 + (i + maze_size//2)*32 + 8}

    f {1 + (i + maze_size//2)*32 + 16} {2 + (i + maze_size//2)*32 + 16} {3 + (i + maze_size//2)*32 + 16}
    f {1 + (i + maze_size//2)*32 + 16} {3 + (i + maze_size//2)*32 + 16} {4 + (i + maze_size//2)*32 + 16}
    f {5 + (i + maze_size//2)*32 + 16} {6 + (i + maze_size//2)*32 + 16} {7 + (i + maze_size//2)*32 + 16}
    f {5 + (i + maze_size//2)*32 + 16} {7 + (i + maze_size//2)*32 + 16} {8 + (i + maze_size//2)*32 + 16}
    f {1 + (i + maze_size//2)*32 + 16} {5 + (i + maze_size//2)*32 + 16} {6 + (i + maze_size//2)*32 + 16}
    f {1 + (i + maze_size//2)*32 + 16} {6 + (i + maze_size//2)*32 + 16} {2 + (i + maze_size//2)*32 + 16}
    f {2 + (i + maze_size//2)*32 + 16} {6 + (i + maze_size//2)*32 + 16} {7 + (i + maze_size//2)*32 + 16}
    f {2 + (i + maze_size//2)*32 + 16} {7 + (i + maze_size//2)*32 + 16} {3 + (i + maze_size//2)*32 + 16}
    f {3 + (i + maze_size//2)*32 + 16} {7 + (i + maze_size//2)*32 + 16} {8 + (i + maze_size//2)*32 + 16}
    f {3 + (i + maze_size//2)*32 + 16} {8 + (i + maze_size//2)*32 + 16} {4 + (i + maze_size//2)*32 + 16}
    f {4 + (i + maze_size//2)*32 + 16} {8 + (i + maze_size//2)*32 + 16} {5 + (i + maze_size//2)*32 + 16}
    f {4 + (i + maze_size//2)*32 + 16} {5 + (i + maze_size//2)*32 + 16} {1 + (i + maze_size//2)*32 + 16}

    f {3 + (i + maze_size//2)*32 + 16} {2 + (i + maze_size//2)*32 + 16} {1 + (i + maze_size//2)*32 + 16}
    f {4 + (i + maze_size//2)*32 + 16} {3 + (i + maze_size//2)*32 + 16} {1 + (i + maze_size//2)*32 + 16}
    f {7 + (i + maze_size//2)*32 + 16} {6 + (i + maze_size//2)*32 + 16} {5 + (i + maze_size//2)*32 + 16}
    f {8 + (i + maze_size//2)*32 + 16} {7 + (i + maze_size//2)*32 + 16} {5 + (i + maze_size//2)*32 + 16}
    f {6 + (i + maze_size//2)*32 + 16} {5 + (i + maze_size//2)*32 + 16} {1 + (i + maze_size//2)*32 + 16}
    f {2 + (i + maze_size//2)*32 + 16} {6 + (i + maze_size//2)*32 + 16} {1 + (i + maze_size//2)*32 + 16}
    f {7 + (i + maze_size//2)*32 + 16} {6 + (i + maze_size//2)*32 + 16} {2 + (i + maze_size//2)*32 + 16}
    f {3 + (i + maze_size//2)*32 + 16} {7 + (i + maze_size//2)*32 + 16} {2 + (i + maze_size//2)*32 + 16}
    f {8 + (i + maze_size//2)*32 + 16} {7 + (i + maze_size//2)*32 + 16} {3 + (i + maze_size//2)*32 + 16}
    f {4 + (i + maze_size//2)*32 + 16} {8 + (i + maze_size//2)*32 + 16} {3 + (i + maze_size//2)*32 + 16}
    f {5 + (i + maze_size//2)*32 + 16} {8 + (i + maze_size//2)*32 + 16} {4 + (i + maze_size//2)*32 + 16}
    f {1 + (i + maze_size//2)*32 + 16} {5 + (i + maze_size//2)*32 + 16} {4 + (i + maze_size//2)*32 + 16}

    f {1 + (i + maze_size//2)*32 + 24} {2 + (i + maze_size//2)*32 + 24} {3 + (i + maze_size//2)*32 + 24}
    f {1 + (i + maze_size//2)*32 + 24} {3 + (i + maze_size//2)*32 + 24} {4 + (i + maze_size//2)*32 + 24}
    f {5 + (i + maze_size//2)*32 + 24} {6 + (i + maze_size//2)*32 + 24} {7 + (i + maze_size//2)*32 + 24}
    f {5 + (i + maze_size//2)*32 + 24} {7 + (i + maze_size//2)*32 + 24} {8 + (i + maze_size//2)*32 + 24}
    f {1 + (i + maze_size//2)*32 + 24} {5 + (i + maze_size//2)*32 + 24} {6 + (i + maze_size//2)*32 + 24}
    f {1 + (i + maze_size//2)*32 + 24} {6 + (i + maze_size//2)*32 + 24} {2 + (i + maze_size//2)*32 + 24}
    f {2 + (i + maze_size//2)*32 + 24} {6 + (i + maze_size//2)*32 + 24} {7 + (i + maze_size//2)*32 + 24}
    f {2 + (i + maze_size//2)*32 + 24} {7 + (i + maze_size//2)*32 + 24} {3 + (i + maze_size//2)*32 + 24}
    f {3 + (i + maze_size//2)*32 + 24} {7 + (i + maze_size//2)*32 + 24} {8 + (i + maze_size//2)*32 + 24}
    f {3 + (i + maze_size//2)*32 + 24} {8 + (i + maze_size//2)*32 + 24} {4 + (i + maze_size//2)*32 + 24}
    f {4 + (i + maze_size//2)*32 + 24} {8 + (i + maze_size//2)*32 + 24} {5 + (i + maze_size//2)*32 + 24}
    f {4 + (i + maze_size//2)*32 + 24} {5 + (i + maze_size//2)*32 + 24} {1 + (i + maze_size//2)*32 + 24}

    f {3 + (i + maze_size//2)*32 + 24} {2 + (i + maze_size//2)*32 + 24} {1 + (i + maze_size//2)*32 + 24}
    f {4 + (i + maze_size//2)*32 + 24} {3 + (i + maze_size//2)*32 + 24} {1 + (i + maze_size//2)*32 + 24}
    f {7 + (i + maze_size//2)*32 + 24} {6 + (i + maze_size//2)*32 + 24} {5 + (i + maze_size//2)*32 + 24}
    f {8 + (i + maze_size//2)*32 + 24} {7 + (i + maze_size//2)*32 + 24} {5 + (i + maze_size//2)*32 + 24}
    f {6 + (i + maze_size//2)*32 + 24} {5 + (i + maze_size//2)*32 + 24} {1 + (i + maze_size//2)*32 + 24}
    f {2 + (i + maze_size//2)*32 + 24} {6 + (i + maze_size//2)*32 + 24} {1 + (i + maze_size//2)*32 + 24}
    f {7 + (i + maze_size//2)*32 + 24} {6 + (i + maze_size//2)*32 + 24} {2 + (i + maze_size//2)*32 + 24}
    f {3 + (i + maze_size//2)*32 + 24} {7 + (i + maze_size//2)*32 + 24} {2 + (i + maze_size//2)*32 + 24}
    f {8 + (i + maze_size//2)*32 + 24} {7 + (i + maze_size//2)*32 + 24} {3 + (i + maze_size//2)*32 + 24}
    f {4 + (i + maze_size//2)*32 + 24} {8 + (i + maze_size//2)*32 + 24} {3 + (i + maze_size//2)*32 + 24}
    f {5 + (i + maze_size//2)*32 + 24} {8 + (i + maze_size//2)*32 + 24} {4 + (i + maze_size//2)*32 + 24}
    f {1 + (i + maze_size//2)*32 + 24} {5 + (i + maze_size//2)*32 + 24} {4 + (i + maze_size//2)*32 + 24}\n'''
            
        # Add internal walls to maze.

        face_count = maze_size*32

        maze_walls = np.zeros([maze_size*2 - 1,maze_size*2 - 1])

        for i in range(1, maze_size*2 - 1, 2):
            for j in range(1, maze_size*2 - 1, 2):
                maze_walls[i,j] = 2

        for i in range(maze_size*2 - 1):
            if i%2 == 0:
                j_start = 1
            else:
                j_start = 0

            for j in range(j_start, maze_size*2 - 1, 2):
                    if np.random.random() < wall_rate:
                        maze_walls[i,j] = 1

                        if j_start:
                            obj_file += f'''
    v {0.05-((maze_size-1)//2) + j//2} {(maze_size-1)//2 - i//2} 0
    v {0.05-((maze_size-1)//2) + j//2} {(maze_size-1)//2 - i//2+1} 0
    v {0.05-((maze_size-1)//2) + j//2} {(maze_size-1)//2 - i//2+1} 0.5
    v {0.05-((maze_size-1)//2) + j//2} {(maze_size-1)//2 - i//2} 0.5
    v {-0.05-((maze_size-1)//2) + j//2} {(maze_size-1)//2 - i//2} 0
    v {-0.05-((maze_size-1)//2) + j//2} {(maze_size-1)//2 - i//2+1} 0
    v {-0.05-((maze_size-1)//2) + j//2} {(maze_size-1)//2 - i//2+1} 0.5
    v {-0.05-((maze_size-1)//2) + j//2} {(maze_size-1)//2 - i//2} 0.5

    f {1 + face_count} {2 + face_count} {3 + face_count}
    f {1 + face_count} {3 + face_count} {4 + face_count}
    f {5 + face_count} {6 + face_count} {7 + face_count}
    f {5 + face_count} {7 + face_count} {8 + face_count}
    f {1 + face_count} {5 + face_count} {6 + face_count}
    f {1 + face_count} {6 + face_count} {2 + face_count}
    f {2 + face_count} {6 + face_count} {7 + face_count}
    f {2 + face_count} {7 + face_count} {3 + face_count}
    f {3 + face_count} {7 + face_count} {8 + face_count}
    f {3 + face_count} {8 + face_count} {4 + face_count}
    f {4 + face_count} {8 + face_count} {5 + face_count}
    f {4 + face_count} {5 + face_count} {1 + face_count}

    f {3 + face_count} {2 + face_count} {1 + face_count}
    f {4 + face_count} {3 + face_count} {1 + face_count}
    f {7 + face_count} {6 + face_count} {5 + face_count}
    f {8 + face_count} {7 + face_count} {5 + face_count}
    f {6 + face_count} {5 + face_count} {1 + face_count}
    f {2 + face_count} {6 + face_count} {1 + face_count}
    f {7 + face_count} {6 + face_count} {2 + face_count}
    f {3 + face_count} {7 + face_count} {2 + face_count}
    f {8 + face_count} {7 + face_count} {3 + face_count}
    f {4 + face_count} {8 + face_count} {3 + face_count}
    f {5 + face_count} {8 + face_count} {4 + face_count}
    f {1 + face_count} {5 + face_count} {4 + face_count}\n'''
                            
                        else:
                            obj_file += f'''
    v {0.05-((maze_size-1)//2) + j//2} {(maze_size-1)//2 - i//2} 0
    v {0.05-((maze_size-1)//2) + j//2-1} {(maze_size-1)//2 - i//2} 0
    v {0.05-((maze_size-1)//2) + j//2-1} {(maze_size-1)//2 - i//2} 0.5
    v {0.05-((maze_size-1)//2) + j//2} {(maze_size-1)//2 - i//2} 0.5
    v {-0.05-((maze_size-1)//2) + j//2} {(maze_size-1)//2 - i//2} 0
    v {-0.05-((maze_size-1)//2) + j//2-1} {(maze_size-1)//2 - i//2} 0
    v {-0.05-((maze_size-1)//2) + j//2-1} {(maze_size-1)//2 - i//2} 0.5
    v {-0.05-((maze_size-1)//2) + j//2} {(maze_size-1)//2 - i//2} 0.5

    f {1 + face_count} {2 + face_count} {3 + face_count}
    f {1 + face_count} {3 + face_count} {4 + face_count}
    f {5 + face_count} {6 + face_count} {7 + face_count}
    f {5 + face_count} {7 + face_count} {8 + face_count}
    f {1 + face_count} {5 + face_count} {6 + face_count}
    f {1 + face_count} {6 + face_count} {2 + face_count}
    f {2 + face_count} {6 + face_count} {7 + face_count}
    f {2 + face_count} {7 + face_count} {3 + face_count}
    f {3 + face_count} {7 + face_count} {8 + face_count}
    f {3 + face_count} {8 + face_count} {4 + face_count}
    f {4 + face_count} {8 + face_count} {5 + face_count}
    f {4 + face_count} {5 + face_count} {1 + face_count}

    f {3 + face_count} {2 + face_count} {1 + face_count}
    f {4 + face_count} {3 + face_count} {1 + face_count}
    f {7 + face_count} {6 + face_count} {5 + face_count}
    f {8 + face_count} {7 + face_count} {5 + face_count}
    f {6 + face_count} {5 + face_count} {1 + face_count}
    f {2 + face_count} {6 + face_count} {1 + face_count}
    f {7 + face_count} {6 + face_count} {2 + face_count}
    f {3 + face_count} {7 + face_count} {2 + face_count}
    f {8 + face_count} {7 + face_count} {3 + face_count}
    f {4 + face_count} {8 + face_count} {3 + face_count}
    f {5 + face_count} {8 + face_count} {4 + face_count}
    f {1 + face_count} {5 + face_count} {4 + face_count}\n'''


                        face_count += 8    

    # print(maze_walls)
        
    with open("pybullet/models/scenes/cl/meshes/cl.obj", "w") as f:
        f.write(obj_file)

    return np.flip(maze_walls,0) if show_walls else -1