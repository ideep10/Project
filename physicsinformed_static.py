import torch
import numpy as np
import matplotlib.pyplot as plt
import utilities
import math

class PhysicsInformedBarModel:
    """A class used for the definition of Physics Informed Models for one dimensional bars."""

    def __init__(self, E, nu, bh, u0, dist_load):
        """Construct a PhysicsInformedBar model"""

        self.E = E*1
        self.nu = nu
        self.bh = bh
        self.u0 = u0
        self.dist_load = dist_load
        self.model = utilities.build_model(1,[40,40],1)
        self.differential_equation_loss_history = None
        self.boundary_condition_loss_history = None
        self.total_loss_history = None
        self.optimizer = None
        
        #Intialization of l0,l1,l2,D1,D2,mu,L,b,h
        self.l = self.bh/math.sqrt(10.4-(15.6*self.nu))
        self.h = 2*self.l
        self.L = 20*self.h
        self.b = 2*self.h
        self.I = self.b * (self.h**3)/12
        self.A = self.b * self.h
        self.mu = self.E/(2*(1+self.nu))
        self.D1 = self.E*self.I + self.mu*self.A*(53*(self.l**2)/15)
        self.D2 = self.mu*self.I*(14*(self.l**2)/5)
        
        #Genrating x for training
        self.x = utilities.generate_grid_1d(self.L,samples=100)

    def get_displacements(self, x):
        """Get displacements."""

        u = self.model(x)   # predict

        return u

    def costFunction(self, x, u_pred):
        """Compute the cost function."""
        # Differential equation loss (f)
        differential_equation_loss = self.D1 * utilities.get_derivative(u_pred,x,4) - self.D2 * utilities.get_derivative(u_pred,x,6)-self.dist_load(x)
        differential_equation_loss = torch.sum(differential_equation_loss ** 2).view(1)
        
        #Defining V,M,Q
        V = self.D1*utilities.get_derivative(u_pred,x,3) - self.D2*utilities.get_derivative(u_pred,x,5)
        M = self.D1*utilities.get_derivative(u_pred,x,2) - self.D2*utilities.get_derivative(u_pred,x,4)
        Q = self.D2*utilities.get_derivative(u_pred,x,3)

        # Boundary condition loss initialization
        boundary_condition_loss = 0
        # Sum over dirichlet boundary condition losses
        boundary_condition_loss += (u_pred[0] - self.u0[0]) ** 2
        boundary_condition_loss += (utilities.get_derivative(u_pred,x,1)[0] - 0) ** 2
        boundary_condition_loss += (V[-1] - 100*(10**-6)) ** 2
        boundary_condition_loss += (M[-1] - 0) ** 2
        boundary_condition_loss += (utilities.get_derivative(u_pred,x,3)[0] - 0) ** 2
        boundary_condition_loss += (utilities.get_derivative(u_pred,x,3)[-1] - 0) ** 2

        return differential_equation_loss, boundary_condition_loss

    def closure(self):
        """Calculation of training error and gradient"""
        self.optimizer.zero_grad()
        u_pred = self.get_displacements(self.x)
        loss = self.costFunction(self.x, u_pred)
        loss = loss[0] + loss[1] 
        loss.backward(retain_graph=True)
        return loss

    def train(self, epochs, optimizer='Adam', **kwargs):
        """Train the model."""

        # Set optimizer
        if optimizer=='Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), **kwargs)
        
        elif optimizer=='LBFGS':
            self.optimizer = torch.optim.LBFGS(self.model.parameters(), **kwargs)

        # Initialize history arrays
        self.differential_equation_loss_history = np.zeros(epochs)
        self.boundary_condition_loss_history = np.zeros(epochs)
        self.total_loss_history = np.zeros(epochs)

        # Training loop
        for i in range(epochs):
            # Predict displacements
            u_pred = self.get_displacements(self.x)

            # Cost function calculation
            differential_equation_loss, boundary_condition_loss = self.costFunction(self.x, u_pred)

            # Total loss
            total_loss = differential_equation_loss + boundary_condition_loss

            # Add energy values to history
            self.differential_equation_loss_history[i] += differential_equation_loss
            self.boundary_condition_loss_history[i] += boundary_condition_loss
            self.total_loss_history[i] += total_loss

            # Print training state
            self.print_training_state(i, epochs)

            # Update parameters (Neural network train)
            self.optimizer.step(self.closure)

    def print_training_state(self, epoch, epochs, print_every=100):
        """Print the loss values of the current epoch in a training loop."""

        if epoch == 0 or epoch == (epochs - 1) or epoch % print_every == 0 or print_every == 'all':
            # Prepare string
            string = "Epoch: {}/{}\t\tDifferential equation loss = {:1f}\t\tBoundary condition loss = {:1f}\t\tTotal loss = {:2f}"

            # Format string and print
            print(string.format(epoch, epochs - 1, self.differential_equation_loss_history[epoch],
                                self.boundary_condition_loss_history[epoch], self.total_loss_history[epoch]))

    def plot_training_history(self, yscale='log'):
        """Plot the training history."""

        # Set up plot
        fig, ax = plt.subplots(figsize=(4,3))
        ax.set_title("Cost function history")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Cost function C")
        plt.yscale(yscale)

        # Plot data
        ax.plot(self.total_loss_history, 'k', linewidth=2, label="Total cost")
        ax.plot(self.differential_equation_loss_history, color='silver', linestyle='--', linewidth=2, label="Differential equation loss")
        ax.plot(self.boundary_condition_loss_history, color='r', linestyle='-.', linewidth=2, label="Boundary condition loss")
        
        ax.legend()
        fig.tight_layout()
        plt.show() 
        
        
