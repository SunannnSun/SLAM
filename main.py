# Pratik Chaudhari (pratikac@seas.upenn.edu)

import click, tqdm, random

from slam import *

def run_dynamics_step(src_dir, log_dir, idx, split, t0=0, draw_fig=False):
    """
    This function is for you to test your dynamics update step. It will create
    two figures after you run it. The first one is the robot location trajectory
    using odometry information obtained form the lidar. The second is the trajectory
    using the PF with a very small dynamics noise. The two figures should look similar.
    """
    slam = slam_t(Q=1e-8*np.eye(3))
    slam.read_data(src_dir, idx, split)

    # trajectory using odometry (xy and yaw) in the lidar data
    d = slam.lidar   # Array of Dictionary
    xyth = []
    for p in d:      # Element p in array d is a dictoinary whose 'xyth' entry contains the pose of robot
        xyth.append([p['xyth'][0], p['xyth'][1],p['xyth'][2]])    # Append a list of x y theta at eacn time stamp to a list
    xyth = np.array(xyth)

    plt.figure(1); plt.clf();
    plt.title('Trajectory using onboard odometry')
    plt.plot(xyth[:,0], xyth[:,1])
    logging.info('> Saving odometry plot in '+os.path.join(log_dir, 'odometry_%s_%02d.jpg'%(split, idx)))
    plt.savefig(os.path.join(log_dir, 'odometry_%s_%02d.jpg'%(split, idx)))

    # dynamics propagation using particle filter
    # n: number of particles, w: weights, p: particles (3 dimensions, n particles)
    # S covariance of the xyth location
    # particles are initialized at the first xyth given by the lidar
    # for checking in this function
    n = 3
    w = np.ones(n)/float(n)
    p = np.zeros((3,n), dtype=np.float64)
    slam.init_particles(n,p,w)
    slam.p[:,0] = deepcopy(slam.lidar[0]['xyth'])
    print('> Running prediction')
    t0 = 0
    T = len(d)
    ps = deepcopy(slam.p)   # maintains all particles across all time steps
    plt.figure(2); plt.clf();
    ax = plt.subplot(111)
    for t in tqdm.tqdm(range(t0+1,T)):  # from t=1 to T; make sense since we neet to get control from t=0 to t=1
        slam.dynamics_step(t)
        ps = np.hstack((ps, slam.p))

        if draw_fig:
            ax.clear()
            ax.plot(slam.p[0], slam.p[0], '*r')
            plt.title('Particles %03d'%t)
            plt.draw()
            plt.pause(0.01)

    plt.plot(ps[0], ps[1], '*c')
    plt.title('Trajectory using PF')
    logging.info('> Saving plot in '+os.path.join(log_dir, 'dynamics_only_%s_%02d.jpg'%(split, idx)))
    plt.savefig(os.path.join(log_dir, 'dynamics_only_%s_%02d.jpg'%(split, idx)))

def run_observation_step(src_dir, log_dir, idx, split, is_online=False):
    """
    This function is for you to debug your observation update step
    It will create three particles np.array([[0.2, 2, 3],[0.4, 2, 5],[0.1, 2.7, 4]])
    * Note that the particle array has the shape 3 x num_particles so
    the first particle is at [x=0.2, y=0.4, z=0.1]
    This function will build the first map and update the 3 particles for one time step.
    After running this function, you should get that the weight of the second particle is the largest since it is the closest to the origin [0, 0, 0]
    """
    slam = slam_t(resolution=0.05)
    slam.read_data(src_dir, idx, split)

    # t=0 sets up the map using the yaw of the lidar, do not use yaw for
    # other timestep
    # initialize the particles at the location of the lidar so that we have some
    # occupied cells in the map to calculate the observation update in the next step
    t0 = 0
    xyth = slam.lidar[t0]['xyth']
    xyth[2] = slam.lidar[t0]['rpy'][2]
    logging.debug('> Initializing 1 particle at: {}'.format(xyth))
    slam.init_particles(n=1,p=xyth.reshape((3,1)),w=np.array([1]))

    slam.observation_step(t=0)
    logging.info('> Particles\n: {}'.format(slam.p))
    logging.info('> Weights: {}'.format(slam.w))

    # reinitialize particles, this is the real test
    logging.info('\n')
    n = 3
    w = np.ones(n)/float(n)
    p = np.array([[2, 0.2, 3],[2, 0.4, 5],[2.7, 0.1, 4]])
    slam.init_particles(n, p, w)

    slam.observation_step(t=1)
    logging.info('> Particles\n: {}'.format(slam.p))
    logging.info('> Weights: {}'.format(slam.w))

def run_slam(src_dir, log_dir, idx, split):
    """
    This function runs slam. We will initialize the slam just like the observation_step
    before taking dynamics and observation updates one by one. You should initialize
    the slam with n=100 particles, you will also have to change the dynamics noise to
    be something larger than the very small value we picked in run_dynamics_step function
    above.
    """
    slam = slam_t(resolution=0.05, Q=np.diag([2e-1,2e-1,1e-1]))
    slam.read_data(src_dir, idx, split)
    T = len(slam.lidar)
    print(T)

    # again initialize the map to enable calculation of the observation logp in
    # future steps, this time we want to be more careful and initialize with the
    # correct lidar scan. First find the time t0 around which we have both LiDAR
    # data and joint data
    t0 = 0
    xyth = slam.lidar[t0]['xyth']
    xyth[2] = slam.lidar[t0]['rpy'][2]
    logging.debug('> Initializing 1 particle at: {}'.format(xyth))
    slam.init_particles(n=1,p=xyth.reshape((3,1)),w=np.array([1]))

    slam.observation_step(t=0)


    n = 5
    p = np.zeros((3, n))
    p_low = np.array([slam.map.xmin, slam.map.ymin, -np.pi])
    p_max = np.array([slam.map.xmax, slam.map.ymax,  np.pi])
    for i in range(3):
        p[i, :] = np.random.uniform(p_low[i], p_max[i], n)
    
    p[:, 0] = np.array([0, 0, 0])

    w = np.ones(n)/float(n)
    slam.init_particles(n, p, w)

    # T = 1

    d = slam.lidar   # Array of Dictionary
    xyth = []
    for p in d:      # Element p in array d is a dictoinary whose 'xyth' entry contains the pose of robot
        xyth.append([p['xyth'][0], p['xyth'][1],p['xyth'][2]])    # Append a list of x y theta at eacn time stamp to a list
    xyth = np.array(xyth)

    estimated_p = np.zeros((3, T))
    # slam, save data to be plotted later
    for t in np.arange(1, T):
        print("Current iteration: %i" %(t))
        slam.dynamics_step(t)   
        estimated_p[:, t] = slam.observation_step(t)



    fig, ax = plt.subplots()
    ax.scatter(estimated_p[0, :], estimated_p[1, :], c='b', s=1)
    ax.scatter(xyth[:, 0], xyth[:, 1], c='r', s=0.5)
    ax.set_title("Odometry vs. Particle Filter")
    ax.legend(['Odometry', 'Particle Filter'])
    


    fig, ax = plt.subplots()
    occupied_cells = slam.map.cells
    occupied_idx = np.argwhere(occupied_cells==1).T

    occupied_xy = np.zeros((2, occupied_idx.shape[1]))
    occupied_xy[0, :] = occupied_idx[0, :] * slam.map.resolution + slam.map.xmin
    occupied_xy[1, :] = occupied_idx[1, :] * slam.map.resolution + slam.map.ymin
    ax.scatter(occupied_xy[0, :], occupied_xy[1, :], c='k')
    ax.set_title("Finalized Binarized Occupancy Map")
    

    plt.show()
    # plt.figure(1); plt.clf();
    # plt.title('Trajectory using particle filter')
    # plt.plot(estimated_p[:,0], estimated_p[:,1])
    # logging.info('> Saving particle filter plot in '+os.path.join(log_dir, 'particle_%s_%02d.jpg'%(split, idx)))
    # plt.savefig(os.path.join(log_dir, 'particle_%s_%02d.jpg'%(split, idx)))




@click.command()
@click.option('--src_dir', default='./', help='data directory', type=str)
@click.option('--log_dir', default='logs', help='directory to save logs', type=str)
@click.option('--idx', default='0', help='dataset number', type=int)
@click.option('--split', default='train', help='train/test split', type=str)
@click.option('--mode', default='slam',
              help='choices: dynamics OR observation OR slam', type=str)
def main(src_dir, log_dir, idx, split, mode):
    # Run python main.py --help to see how to provide command line arguments

    if not mode in ['slam', 'dynamics', 'observation']:
        raise ValueError('Unknown argument --mode %s'%mode)
        sys.exit(1)

    np.random.seed(42)
    random.seed(42)

    if mode == 'dynamics':
        run_dynamics_step(src_dir, log_dir, idx, split)
        sys.exit(0)
    elif mode == 'observation':
        run_observation_step(src_dir, log_dir, idx, split)
        sys.exit(0)
    else:
        p = run_slam(src_dir, log_dir, idx, split)
        return p

if __name__=='__main__':
    main()