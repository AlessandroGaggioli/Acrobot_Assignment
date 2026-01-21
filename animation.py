#Animation 

from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
import math
import parameters as par


def animate_double_pendolum(xx_star, xx_ref, dt, title = 'Pendulum animation'):
      
      #print("deg:",xx_ref[0, 1]) #theta1, theta2, theta1dot, theta2dot, TT=1
      #deg->rad
      xx_star = xx_star*(math.pi/180)
      xx_ref = xx_ref*(math.pi/180)
      #print("rad:",xx_ref[0, 1])

      """
      Animates the pendolum dynamics
      input parameters:
            - Optimal state trajectory xx_star
            - Reference trajectory xx_ref
            - Sampling time dt
      oputput arguments:
            None
      """

      TT = xx_star.shape[1] #number of columns

      # Set up the figure and axis for the animation
      fig, ax = plt.subplots(figsize=(10,10))
      #ax.set_xlim(-((par.l1 + par.l2)*1.05), ((par.l1 + par.l2)*1.05))  #limits based on length of the links
      #ax.set_ylim(-((par.l1 + par.l2)*1.05), ((par.l1 + par.l2)*1.05))
      #ax.set_aspect('equal')

      ax.set_xlim(-1.0,1.0) 
      ax.set_ylim(-((par.l1 + par.l2)*1.05), 0.5)
      ax.set_aspect('equal')

      # Plot elements
      # solid line ('o-'), dashed line ('o--'), lw=line width
      pendulum_line, = ax.plot([], [], 'o-', lw=3, color="blue", label="Optimal Path")
      reference_line, = ax.plot([], [], 'o--', lw=2, color="green", label="Reference Path")
      time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
      ax.legend()
      ax.set_title(title)
      ax.set_xlabel("X position")
      ax.set_ylabel("Y position")
      plt.grid(True)
      

      # Initial setup function for the animation
      def init():
            pendulum_line.set_data([], []) # since [] we are clearing the plot
            reference_line.set_data([], [])
            time_text.set_text('')  # clear time value
            return pendulum_line, reference_line, time_text

      # Update function for each frame of the animation
      def update(frame):
            # Pendulum position (optimal solution)

            """print("sin_1", np.sin(xx_ref[0, frame]))
            print("cos_1", np.cos(xx_ref[0, frame]))
            print("sin_2", np.sin(xx_ref[1, frame]))
            print("cos_2", np.cos(xx_ref[1, frame]))
            """
            x1_opt = np.sin(xx_star[0, frame])  # assuming xx_star[0] is angle theta1
            y1_opt = -np.cos(xx_star[0, frame])
            x2_opt = x1_opt + np.sin(xx_star[1, frame])  # assuming xx_star[1] is angle theta2
            y2_opt = y1_opt - np.cos(xx_star[1, frame])

            # Reference position
            x1_ref = np.sin(xx_ref[0, frame])
            y1_ref = -np.cos(xx_ref[0, frame])
            x2_ref = x1_ref + np.sin(xx_ref[1, frame])
            y2_ref = y1_ref - np.cos(xx_ref[1, frame])


            # Update pendulum line
            pendulum_line.set_data([0, x1_opt,x2_opt], [0, y1_opt,y2_opt]) # ([x-pivot, x-coordinate], .. )
            reference_line.set_data([0, x1_ref,x2_ref], [0, y1_ref,y2_ref])
            
            # Update time text
            time_text.set_text(f'time = {frame*dt:.2f}s')

            return pendulum_line, reference_line, time_text

      # Create the animation
      ani = FuncAnimation(fig, update, frames=TT, init_func=init, blit=True, interval=par.dt*1000)

      # Display the animation
      plt.show()