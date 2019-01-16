/* Trials with CHOMP.
 *
 * Copyright (C) 2014 Roland Philippsen. All rights reserved.
 *
 * BSD license:
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of
 *    contributors to this software may be used to endorse or promote
 *    products derived from this software without specific prior written
 *    permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR THE CONTRIBUTORS TO THIS SOFTWARE BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
   \file pprrr2d.cpp
   
   Interactive trials with CHOMP for 3-DOF planar arms mounted on a
   2-DOF planar vehicle moving holonomously in the plane.
   
   Otherwise pretty much the same as pp2d.cpp, except that we actually
   sum up the obstacle cost (and gradient) for a selected few body
   points instead of just the center of the base. There is a fixed
   start and goal configuration, and you can drag a circular obstacle
   around to see how the CHOMP algorithm reacts to that.  Some of the
   computations involve guesswork, for instance how best to compute
   velocities, so a simple first-order scheme has been used.  This
   appears to produce some unwanted drift of waypoints from the start
   configuration to the end configuration.  Parameters could also be
   tuned a bit better.  Other than that, it works pretty nicely.
*/

#include "gfx.hpp"
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <stdlib.h>
#include <sys/time.h>
#include <err.h>
#include <chrono>
#include <stdlib.h>

#include <SerialPort.h>
#include <SerialStream.h>
#include <cstdlib>
#include <fstream>
#include <unistd.h>
#include <sys/poll.h>

#include <stdio.h>
      #include <fcntl.h>   /* File Control Definitions           */
      #include <termios.h> /* POSIX Terminal Control Definitions */
      #include <unistd.h>  /* UNIX Standard Definitions      */ 
      #include <errno.h>   /* ERROR Number Definitions           */

       #include <arpa/inet.h>

#define INTMATH 0
#define loopcountvar 1000

typedef Eigen::VectorXd Vector;
typedef Eigen::MatrixXd Matrix;
typedef Eigen::Isometry3d Transform;

using namespace std;
Eigen::IOFormat CleanFmt(2, 0, ", ", "\n", "[", "]");
Eigen::IOFormat HeavyFmt(Eigen::FullPrecision, 0, ", ", ";\n", "[", "]", "[", "]");

//////////////////////////////////////////////////
// trajectory etc

Vector xi;			// the trajectory (q_1, q_2, ...q_n)
Vector qs;			// the start config a.k.a. q_0
Vector qe;			// the end config a.k.a. q_(n+1)
Eigen::VectorXi xi2;
Eigen::VectorXi qs2;
Eigen::VectorXi qe2;

static size_t const obs_dim(3); // x,y,R
static size_t const nq (20);	// number of q stacked into xi
static size_t const cdim (5);	// dimension of config space
static size_t const xidim (nq * cdim); // dimension of trajectory, xidim = nq * cdim
static double const dt (1.0);	       // time step
static double const eta (100.0); // >= 1, regularization factor for gradient descent
static double const lambda (1.0); // weight of smoothness objective
static int const dt2 (1);        // time step
static int const eta2 (100); // >= 1, regularization factor for gradient descent
static int const lambda2 (1); // weight of smoothness objective
static int const scale(1e3);
int globalflag(0);
//////////////////////////////////////////////////
// gradient descent etc

Matrix AA;			// metric
Vector  bb;			// acceleration bias for start and end config
Matrix Ainv;			// inverse of AA
Matrix obs;                     // Matrix containninging all obstacles, each column is (x,y,R) of the obstacle

Eigen::MatrixXi AA2;
Eigen::VectorXi bb2;
Eigen::MatrixXi Ainv2;
//Matrix Ainv2;
Eigen::MatrixXi obs2;


//////////////////////////////////////////////////
// gui stuff

enum { PAUSE, STEP, RUN } state;


static Vector grab_offset (3);
static int grabbed(-1);
int countvar = 0;
std::chrono::duration<double> elapsed;

//////////////////////////////////////////////////
// robot (one per waypoint)

class Robot
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  Robot ()
    : pos_a_ (3),
      pos_b_ (3),
      pos_c_ (3)
  {
  }
  
  
  Transform frame (size_t node) const
  {
    Transform tf (Transform::Identity());
    switch (node) {
    case 0:
      tf.translation() << position_[0], position_[1], 0.0;
      break;
    case 1:
      tf.translation() << position_[0], position_[1], 0.0;
      //tf.linear() << c2_, -s2_, 0.0, s2_, c2_, 0.0, 0.0, 0.0, 1.0;
      break;
    case 2:
      tf.translation() << pos_a_[0], pos_a_[1], 0.0;
      //tf.linear() << c23_, -s23_, 0.0, s23_, c23_, 0.0, 0.0, 0.0, 1.0;
      break;
    case 3:
      tf.translation() << pos_b_[0], pos_b_[1], 0.0;
      //tf.linear() << c234_, -s234_, 0.0, s234_, c234_, 0.0, 0.0, 0.0, 1.0;
      break;
    default:
      errx (EXIT_FAILURE, "Robot::frame() called on invalid node %zu", node);
    }
    return tf;
  }
    Transform frame2 (size_t node) const
  {
    Transform tf (Transform::Identity());
    switch (node) {
    case 0:
      tf.translation() << position_[0]*1000, position_[1]*1000, 0.0;
      break;
    case 1:
      tf.translation() << position_[0]*1000, position_[1]*1000, 0.0;
      break;
    case 2:
      tf.translation() << pos_a_[0]*1000, pos_a_[1]*1000, 0.0;
      break;
    case 3:
      tf.translation() << pos_b_[0]*1000, pos_b_[1]*1000, 0.0;
      break;
    default:
      errx (EXIT_FAILURE, "Robot::frame() called on invalid node %zu", node);
    }
    return tf;
  }

  
  
  Matrix computeJxo (size_t node, Vector const & gpoint) const
  {
    Matrix Jxo (Matrix::Zero (6, 5));
    switch (node) {
    case 3:
      Jxo (0, 4) = pos_b_[1] - gpoint[1];
      Jxo (1, 4) = gpoint[0] - pos_b_[0];
      Jxo (5, 4) = 1.0;
    case 2:
      Jxo (0, 3) = pos_a_[1] - gpoint[1];
      Jxo (1, 3) = gpoint[0] - pos_a_[0];
      Jxo (5, 3) = 1.0;
    case 1:
      Jxo (0, 2) = position_[1] - gpoint[1];
      Jxo (1, 2) = gpoint[0]    - position_[0];
      Jxo (5, 2) = 1.0;
    case 0:
      Jxo (0, 0) = 1.0;
      Jxo (1, 1) = 1.0;
      break;
    default:
      errx (EXIT_FAILURE, "Robot::computeJxo() called on invalid node %zu", node);
    }
    return Jxo;
  }
  

   Eigen::MatrixXi computeJxo2 (size_t node, Eigen::VectorXi const & gpoint) const
  {
    Eigen::MatrixXi Jxo2 (Eigen::MatrixXi::Zero (6, 5));
    switch (node) {
    case 3:
      Jxo2 (0, 4) = pos_b_[1]*1000 - gpoint[1];
      Jxo2 (1, 4) = gpoint[0] - pos_b_[0]*1000;
      Jxo2 (5, 4) = 1.0*1000;
    case 2:
      Jxo2 (0, 3) = pos_a_[1]*1000 - gpoint[1];
      Jxo2 (1, 3) = gpoint[0] - pos_a_[0]*1000;
      Jxo2 (5, 3) = 1.0*1000;
    case 1:
      Jxo2 (0, 2) = position_[1]*1000 - gpoint[1];
      Jxo2 (1, 2) = gpoint[0]    - position_[0]*1000;
      Jxo2 (5, 2) = 1.0*1000;
    case 0:
      Jxo2 (0, 0) = 1.0*1000;
      Jxo2 (1, 1) = 1.0*1000;
      break;
    default:
      errx (EXIT_FAILURE, "Robot::computeJxo() called on invalid node %zu", node);
    }
    return Jxo2;
  }

  

  void update (Vector const & position)
  {
    if (position.size() != 5) {
      errx (EXIT_FAILURE, "Robot::update(): position has %zu DOF (but needs 5)",
	    (size_t) position.size());
    }
    position_ = position;
    
    c2_ = cos (position_[2]);
    s2_ = sin (position_[2]);
    ac2_ = len_a_ * c2_;
    as2_ = len_a_ * s2_;
    
    q23_ = position_[2] + position_[3];
    c23_ = cos (q23_);
    s23_ = sin (q23_);
    bc23_ = len_b_ * c23_;
    bs23_ = len_b_ * s23_;
    
    q234_ = q23_ + position_[4];
    c234_ = cos (q234_);
    s234_ = sin (q234_);
    cc234_ = len_c_ * c234_;
    cs234_ = len_c_ * s234_;
    
    pos_a_ <<
      position_[0] + ac2_,
      position_[1] + as2_,
      0.0;
    pos_b_ <<
      pos_a_[0] + bc23_,
      pos_a_[1] + bs23_,
      0.0;
    pos_c_ <<
      pos_b_[0] + cc234_,
      pos_b_[1] + cs234_,
      0.0;
  }
  
  
  void draw () const
  {
    // translucent disk for base
    gfx::set_pen (1.0, 0.7, 0.7, 0.7, 0.5);
    gfx::fill_arc (position_[0], position_[1], radius_, 0.0, 2.0 * M_PI);
    
    // thick circle outline for base
    gfx::set_pen (3.0, 0.2, 0.2, 0.2, 1.0);
    gfx::draw_arc (position_[0], position_[1], radius_, 0.0, 2.0 * M_PI);
    
    // thick line for arms
    gfx::set_pen (3.0, 0.2, 0.2, 0.2, 1.0);
    gfx::draw_line (position_[0], position_[1], pos_a_[0], pos_a_[1]);
    gfx::draw_line (pos_a_[0], pos_a_[1], pos_b_[0], pos_b_[1]);
    gfx::draw_line (pos_b_[0], pos_b_[1], pos_c_[0], pos_c_[1]);
  }
  
  static double const radius_;
  static double const len_a_;
  static double const len_b_;
  static double const len_c_;
  
  Vector position_;
  Vector pos_a_;
  Vector pos_b_;
  Vector pos_c_;
  
  double c2_;
  double s2_;
  double c23_;
  double s23_;
  double c234_;
  double s234_;
  double q23_;
  double q234_;
  double ac2_;
  double as2_;
  double bc23_;
  double bs23_;
  double cc234_;
  double cs234_;
};

double const Robot::radius_ (0.5);
double const Robot::len_a_ (0.8);
double const Robot::len_b_ (0.9);
double const Robot::len_c_ (0.9);

Robot rstart;
Robot rend;
vector <Robot> robots;

#define VAR_TO_STR_BIN(x) obj_to_bin((char [sizeof(x)*CHAR_BIT + 1]){""}, &(x), sizeof (x))

char *obj_to_bin(char *dest, void *object, size_t osize) {
  const unsigned char *p = (const unsigned char *) object;
  p += osize;
  char *s = dest;
  while (osize-- > 0) {
    p--;
    unsigned i = CHAR_BIT;
    while (i-- > 0) {
      *s++ = ((*p >> i) & 1) + '0';
    }
  }
  *s = '\0';
  return dest;
}
//////////////////////////////////////////////////

static void update_robots ()
{
   //if(countvar < loopcountvar){
    //auto start = std::chrono::high_resolution_clock::now();

  rstart.update (qs);
  rend.update (qe);
  if (nq != robots.size()) {
    robots.resize (nq);
  }
  for (size_t ii (0); ii < nq; ++ii) {
    robots[ii].update (xi.block (ii * cdim, 0, cdim, 1));
  }
    //auto finish = std::chrono::high_resolution_clock::now();
    //elapsed= finish - start;
//std::cout << "Elapsed time: " << elapsed.count() << " s\n";
  //countvar++;
  //}
  // else{
  //   //std::cout << "Elapsed time: " << elapsed.count()/100 << " s\n";
  //   elapsed = std::chrono::seconds { 0 };
  //   countvar = 0;
  // }
}


static void init_chomp ()
{
  qs.resize (5);
  qs << -5.0, -5.0, M_PI/2, M_PI/2, -M_PI/2;
  xi = Vector::Zero (xidim);
  qe.resize (5);
  qe << 7.0, 7.0, -M_PI/2, -M_PI/2, M_PI/2;

  qs2.resize (5);
  qs2 << -5*scale, -5*scale, (M_PI/2)*scale, (M_PI/2)*scale, -(M_PI/2)*scale;
  xi2 = Eigen::VectorXi::Zero (xidim); //Vector::Zero (xidim);//
  qe2.resize (5);
  qe2 << 7*scale, 7*scale, -(M_PI/2)*scale, -(M_PI/2)*scale, (M_PI/2)*scale; //might have issue with cast to int
  
  //repulsor.point_ << 6.0, 3.0;
  
  // cout << "qs\n" << qs2
  //      << "\nxi\n" << xi2
  //      << "\nqe\n" << qe2 << "\n\n";
  
  AA = Matrix::Zero (xidim, xidim);
  for (size_t ii(0); ii < nq; ++ii) {
    AA.block (cdim * ii, cdim * ii, cdim , cdim) = 2.0 * Matrix::Identity (cdim, cdim);
    if (ii > 0) {
      AA.block (cdim * (ii-1), cdim * ii, cdim , cdim) = -1.0 * Matrix::Identity (cdim, cdim);
      AA.block (cdim * ii, cdim * (ii-1), cdim , cdim) = -1.0 * Matrix::Identity (cdim, cdim);
    }
  }
  AA /= dt * dt * (nq + 1);
  //std::cout << AA.format(CleanFmt) << std::endl;

   AA2 = Eigen::MatrixXi::Zero (xidim, xidim);
  // for (size_t ii(0); ii < nq; ++ii) {
  //   AA2.block (cdim * ii, cdim * ii, cdim , cdim) = 2 * Eigen::MatrixXi::Identity (cdim, cdim);
  //   if (ii > 0) {
  //     AA2.block (cdim * (ii-1), cdim * ii, cdim , cdim) = -1 * Eigen::MatrixXi::Identity (cdim, cdim);
  //     AA2.block (cdim * ii, cdim * (ii-1), cdim , cdim) = -1 * Eigen::MatrixXi::Identity (cdim, cdim);
  //   }
  // }
  // AA2 = AA2 *scale*scale;
  // AA2 /= dt2 * dt2 * (nq + 1);
  AA2= (AA*scale).cast<int>();
  

  bb = Vector::Zero (xidim);
  bb.block (0,            0, cdim, 1) = qs;
  bb.block (xidim - cdim, 0, cdim, 1) = qe;
  bb /= - dt * dt * (nq + 1);
  //std::cout << bb.format(CleanFmt) << std::endl;
  bb2 = Eigen::VectorXi::Zero (xidim);
  bb2.block (0,            0, cdim, 1) = qs2;
  bb2.block (xidim - cdim, 0, cdim, 1) = qe2;
  bb2 /= - (dt2 * dt2 * (nq + 1));
  bb2 = bb2*1000;
  //std::cout << bb2.format(CleanFmt) << std::endl;
  // not needed anyhow
  // double cc (double (qs.transpose() * qs) + double (qe.transpose() * qe));
  // cc /= dt * dt * (nq + 1);
      std::cout << "The matrix m is of size " << bb2.rows() << "x" << bb2.cols() << std::endl; 

    std::cout << bb2.format(CleanFmt) << std::endl;
  Ainv = AA.inverse();
  //std::cout << Ainv.format(CleanFmt) << std::endl;
  Ainv2 = (Ainv).cast<int>();

  
  //std::cout << Ainv.format(CleanFmt) << std::endl;
  //std::cout << ((Ainv2*AA2)/1000000).format(CleanFmt) << std::endl; 
  // cout << "AA\n" << AA
  //      << "\nAinv\n" << Ainv
  //      << "\nbb\n" << bb << "\n\n";
}


static void cb_step ()
{
  state = STEP;
}


static void cb_run ()
{
  if (RUN == state) {
    state = PAUSE;
  }
  else {
    state = RUN;
  }
}


static void cb_jumble ()
{
  for (size_t ii (0); ii < xidim; ++ii) {
    if (ii % 5 < 2) {
      xi[ii] = double (rand()) / (0.1 * numeric_limits<int>::max()) - 5.0;
    }
    else {
      xi[ii] = double (rand()) / (numeric_limits<int>::max() / 2.0 / M_PI) - M_PI;
    }
  }
  update_robots();
}


static void cb_idle ()
{
  if (PAUSE == state) {
    return;
  }
  if (STEP == state) {
    state = PAUSE;
  }
  
  //////////////////////////////////////////////////
  // beginning of "the" CHOMP iteration
  if(countvar < loopcountvar){
  //AA2 scale by 1000*1000 and bb2 scale by 1000
  #ifdef INTMATH
  Eigen::VectorXi nabla_smooth2 (AA2*( ((xi2/1000)) ) +(bb2));//(((AA * xi + bb)*scale*scale).cast<int>());//
  //Eigen::VectorXi nabla_smoothtest ((AA2*( ((xi*1000).cast<int>()) ) )); //scaled by 1000*1000 *( ((xi*1000).cast<int>()) ) 
     //std::cout << "nabla_smooth2 " << xi2.rows() << "x" << nabla_smooth.cols() << std::endl; 
  Eigen::VectorXi const & xidd2 (nabla_smooth2);
  Eigen::VectorXi nabla_obs2 (Eigen::VectorXi::Zero (xidim));
  #else
    Vector nabla_smooth ((AA * xi + bb)); //
//  std::cout << "nabla_smooth " << nabla_smooth.rows() << "x" << nabla_smooth.cols() << std::endl; 
  // std::cout << "nabla_smooth " << (AA*xi).format(HeavyFmt) << std::endl; //*xi
  Vector const & xidd (nabla_smooth); 
  Vector nabla_obs (Vector::Zero (xidim));

  #endif


  // std::cout << "nabla_smoothtest "<< ( AA2*(xi*1000).cast<int>()).format(HeavyFmt) << std::endl; //.cast<double>()/(scale*scale)
// indeed, it is the same in this formulation...
//.format(HeavyFmt)
  
  //#pragma omp parallel for
  int arr[nq];
  for (int i=0; i < nq; i++)
    {
        arr[i] = i;
    }

  random_shuffle(&arr[0], &arr[nq-1]);
  // for (int i=0; i < nq; i++)
  //   {
  //       std::cout << arr[i] << " ";
  //   }
  #ifdef INTMATH
    obs2 = (obs *1000).cast<int>();
  #endif  
  //std::cout << "---------" << std::endl;

  //std::cout << bb.format(CleanFmt) << std::endl;
  auto start = std::chrono::high_resolution_clock::now();

  for (int p = nq-1; -1 < p; --p) {
    int iq = arr[p];
    Vector qd;
    Eigen::VectorXi qd2;
    if (iq == nq - 1) {

      //qd = qe - xi.block (iq * cdim, 0, cdim, 1);
      //qd2 = (qe2 - (xi.block (iq * cdim, 0, cdim, 1)*scale).cast<int>());

      qd2 = (qe2 - (xi2.block (iq * cdim, 0, cdim, 1)/scale));   
      //std::cout << qd.format(CleanFmt) << std::endl;
      //std::cout << qd2.format(CleanFmt) << std::endl;
    }
    else {
      //qd = xi.block ((iq+1) * cdim, 0, cdim, 1) - xi.block (iq * cdim, 0, cdim, 1);
      //qd2 = (xi.block ((iq+1) * cdim, 0, cdim, 1)*scale).cast<int>() - (xi.block (iq * cdim, 0, cdim, 1)*scale).cast<int>();
      
      qd2 = (xi2.block ((iq+1) * cdim, 0, cdim, 1)/scale) - (xi2.block (iq * cdim, 0, cdim, 1)/scale);
    }
  
    for (size_t ib (0); ib < 4; ++ib) { // later: configurable number of body points




      // std::cout << "---------" << std::endl;

      // std::cout << xx.format(CleanFmt) << std::endl;


      #ifdef INTMATH
      //Eigen::VectorXi const xx2 ((xx*scale).cast<int>());
      Eigen::VectorXi const xx2 (robots[iq].frame2(ib).translation().cast<int>());
      Eigen::MatrixXi const JJ2 (robots[iq].computeJxo2 (ib, xx2).block(0, 0, 2, 5)); // XXXX hardcoded indices
      //std::cout << "xx2 "<<  xx2.rows() << "x" << xx2.cols() << xx2 << std::endl;
      //Eigen::MatrixXi const JJ2 ((JJ*scale).cast<int>());
              
     //std::cout << JJ2.format(CleanFmt) << std::endl;
      //std::cout << "---------" << std::endl;
      //std::cout << JJ2(0,3) << std::endl;
      //int x = (JJ2(0,3) >> (8*1)) & 0xff;
      



     //unsigned long n = 175;

//       unsigned char bytes[4];
//      bytes[0] = (n >> 24) && 0xFF;
//       bytes[1] = (n >> 16) && 0xFF;
//       bytes[2] = (n >> 8) && 0xFF;
//       bytes[3] = n && 0xFF;
// printf("%x %x %x %x\n", bytes[0],
//                         bytes[1],
//                        bytes[2],
//                        bytes[3]);  
   //std::cout << "1 "<<bytes[0]<< "2 "<<bytes[1]<< "3 "<<bytes[2]<< "4 "<<bytes[3] << std::endl;
      //std::cout << "JJ2 "<< JJ2.rows() << "x" << JJ2.cols() << std::endl;
       Eigen::VectorXi const xd2 ((JJ2 * qd2)/scale);
      //std::cout << "xd2 "<< xd2.rows() << "x" << xd2.cols() << std::endl;

      double const vel2 ((xd2.norm()));

      #else
      Vector const xx (robots[iq].frame(ib).translation());
      Matrix const JJ (robots[iq].computeJxo (ib, xx) .block (0, 0, 2, 5)); // XXXX hardcoded indices

      Vector const xd (JJ * qd);
      double const vel (xd.norm());//xd.norm());
      #endif

      // Eigen::VectorXi matA(2, 1);
      // matA << 70000, 70000;
      // int const testvala (matA.dot(matA));
      // std::cout << "matA"<<  matA.format(CleanFmt)<< std::endl;
      // std::cout << "test"<< testvala<< std::endl;
       // std::cout << "xd"<<xd.format(CleanFmt) << std::endl;
       // std::cout << "xd2 "<<xd2.format(CleanFmt) << std::endl;
       // std::cout << "vel "<<vel << std::endl;
       // std::cout << "vel2 "<<vel2 << std::endl;


      #ifdef INTMATH
      if (vel2 < 2)  // avoid div by zero further down
        continue;
      Eigen::VectorXi const xdn2 ((xd2 * scale)/ vel2);
      Eigen::VectorXi const xdd2 (JJ2 * xidd2.block (iq * cdim, 0, cdim , 1) );
      Eigen::MatrixXi const prj2 ((scale*Eigen::MatrixXi::Identity (2, 2)) - (xdn2 * xdn2.transpose()/scale));
      //Eigen::VectorXi const kappa2 ((prj2 * xdd2 / (pow(vel2, 2)))); // / pow(vel2, 2)
      //std::cout << "kappa2 "<< kappa2.format(CleanFmt) << std::endl;
      auto temp =obs2;
      #else
      if (vel < 1.0e-3)
        continue;
      Vector const  xdn (xd / vel);
      Vector const xdd (JJ * xidd.block (iq * cdim, 0, cdim , 1));
      Matrix const prj (Matrix::Identity (2, 2) - xdn * xdn.transpose()); // hardcoded planar case
      Vector const kappa (prj * xdd / pow(vel, 2.0)); // very small could cause issue // 
      //Vector const test  (prj2.cast<double>() * xdd2.cast<double>()/pow(vel2, 2.0)); 
      //auto temp =obs;
      #endif
      

     

          for (int ii = 0; ii < temp.cols(); ii++) {
          
          //cout << obs.size()<< endl;
          //printDimensions(xx);
          //std::cout << obs.format(CleanFmt) << sep;
            //std::cout << "gothere4" << std::endl;

          //Vector delta (xx.block (0, 0, 3, 1) - repulsor.point_);
          //std::cout << "gothere5" << std::endl;
          //std::cout << "xx: "<< xx.format(CleanFmt) << std::endl;
          //std::cout << "xx.new: "<< xx.block (0, 0, 2, 1).format(CleanFmt) << std::endl;
          
            #ifdef INTMATH
            Eigen::VectorXi delta2(xx2.block (0, 0, 2, 1) - obs2.block(0, ii, 2, 1));
            int const dist2(delta2.norm());
            if ((dist2 >= obs2(2, ii)) || (dist2 < 1))
              continue;
            static int const gain2(1);
            //int cost2((scale - (dist2*scale/obs2(2, ii))));// / (3.0*scale));
            //(gain2 * obs2(2, ii))/scale * 
            int temp2 = -gain2 *pow(scale - (dist2*scale)/(obs2(2, ii)), 2.0);// / dist2; 
            delta2 *= temp2;
            delta2 /=scale; 
            nabla_obs2.block(iq * cdim, 0, cdim, 1) += JJ2.transpose()  * vel2 / scale * ((prj2 * delta2)/scale)/scale; // /scale)

            #else
            Vector delta(xx.block (0, 0, 2, 1) - obs.block(0, ii, 2, 1));
            //std::cout << "delta: = " << delta << sep;
            double const dist(delta.norm());
            if ((dist >= obs(2, ii)) || (dist < 1e-9))
              continue;
            static double const gain(1.0);
            double temp = -gain *pow(1.0 - dist /obs(2, ii), 2.0);// / dist;
            //double const cost((1.0- dist / obs(2, ii)));//, 3.0)/3.0); // hardcoded param
            //gain * obs(2, ii) *
            //std::cout << "delta old "<< delta.format(CleanFmt) << std::endl;
            //std::cout << "delta2 old"<< delta2.format(CleanFmt) << std::endl;
            delta *=  temp;
            nabla_obs.block(iq * cdim, 0, cdim, 1) += JJ.transpose()  * vel * ((prj * delta)- (cost2 * kappa2) ); // delta F obslacate close
            //
            #endif

          //std::cout << "delta: "<< delta << std::endl;
          // std::cout << "xx"<< xx.size()<< std::endl;
           //std::cout << "dist: "<< dist << std::endl; 
           //std::cout << "dist2: "<< dist2 << std::endl; 
          // std::cout << " ii): "<< ii << std::endl; 
           //std::cout << " obs(2, ii): "<< obs(2, ii) << std::endl; 
           // std::cout << " obs(2, ii): "<< obs2(2, ii) << std::endl; 

          // std::cout << "cost "<< cost << std::endl;
             // std::cout << "delta "<< delta.format(CleanFmt) << std::endl;
             // std::cout << "delta2 "<< delta2.format(CleanFmt) << std::endl;
             // std::cout << "temp "<< temp << std::endl;
             // std::cout << "temp2 "<< temp2 << std::endl;
            
             // std::cout << "dist "<< dist << std::endl;
             // std::cout << "dist2 "<< dist2 << std::endl;
                   // std::cout << "nabla_smooth "<< nabla_smooth << std::endl;
                   // std::cout << "nabla_smooth2 "<< nabla_smooth2 << std::endl;
                   //  std::cout << "nabla_obs "<< nabla_obs << std::endl;
                   // std::cout << "nabla_obs2 "<< nabla_obs2 << std::endl;
           //std::cout << "cost2 "<< cost2 << std::endl;
           //std::cout << "obs  "<<obs(2, ii) << std::endl;
           //std::cout << "obs2  "<<obs2(2, ii) << std::endl;
          //std::cout << "test "<< test << std::endl;

              if (globalflag <0){
                    // std::cout << "xd"<<xd.format(CleanFmt) << std::endl;
                    // std::cout << "xd2 "<<xd2.format(CleanFmt) << std::endl;
                    // std::cout << "vel "<<vel << std::endl;
                    // std::cout << "vel2 "<<vel2 << std::endl;
                    // std::cout << "xdn "<<xdn.format(CleanFmt) << std::endl;
                    // //std::cout << "xdn2 "<<xdn2.format(CleanFmt) << std::endl;
                    // std::cout << "xdd "<<xdd.format(CleanFmt) << std::endl;
                    // //std::cout << "xdd2 "<<xdd2.format(CleanFmt) << std::endl;
                    // std::cout << "xxb: "<< xx.block (0, 0, 2, 1).format(CleanFmt) << std::endl;
                    // //std::cout << "xx2b: "<< xx2.block (0, 0, 2, 1).format(CleanFmt) << std::endl;
                    // std::cout << "prj "<< prj.format(CleanFmt) << std::endl;
              //std::cout << "prj2 "<< prj2.format(CleanFmt) << std::endl;
              //std::cout << "kappa "<< kappa.format(CleanFmt) << std::endl;
             // std::cout << "kappa2 "<< kappa2.format(CleanFmt) << std::endl;
             // std::cout << "obs  "<<obs(2, ii) << std::endl;
            // std::cout << "obs2  "<<obs2(2, ii) << std::endl;
              // std::cout << "deltanew "<< delta.format(CleanFmt) << std::endl;
              // std::cout << "delta2new "<< delta2.format(CleanFmt) << std::endl;
              // std::cout << "dist "<< dist << std::endl;
              // std::cout << "dist2 "<< dist2 << std::endl;
              // std::cout << "cost "<< cost << std::endl;
              // std::cout << "cost2 "<< cost2 << std::endl;
              // std::cout << "temp "<< temp << std::endl;
              // std::cout << "temp2 "<< temp2 << std::endl;
              //std::cout << "nabla_obs "<< nabla_obs.format(CleanFmt) << std::endl;
              //std::cout << "nabla_obs2 "<< nabla_obs2.format(CleanFmt) << std::endl;

            globalflag +=1;
            }
        }

         
      // Vector delta (xx.block (0, 0, 2, 1) - repulsor.point_);
      // double const dist (delta.norm());
      // static double const maxdist (4.0);
      // if ((dist >= maxdist) || (dist < 1e-9)) {
      //   	continue;
      // }
      // static double const gain (10.0);
      // double const cost (gain * maxdist * pow (1.0 - dist / maxdist, 3.0) / 3.0);
      // delta *= - gain * pow (1.0 - dist / maxdist, 2.0) / dist;
      // nabla_obs.block (iq * cdim, 0, cdim, 1) += JJ.transpose() * vel * (prj * delta - cost * kappa);
    }

  }
  

  
  #ifdef INTMATH
  //Vector dxi2 ((Ainv2 * ((nabla_obs2) + lambda * (nabla_smooth2))).cast<double>()/(scale*scale)); //nabla_smooth2.cast<double>()/(scale*scale)
  //Eigen::VectorXi //
  Eigen::VectorXi dxi2int (Ainv2*((nabla_obs2) +  lambda * (nabla_smooth2)));
  
  //xi -= (dxi2)/ eta;///1000000
  xi2 -= (dxi2int)/eta2; ///(scale*scale) 
  xi = xi2.cast<double>()/(scale*scale);
  // xi2 scale by 1000000     
    //std::cout << "xi "<< (xi).format(CleanFmt) << std::endl;
       //Eigen::VectorXi test (Ainv2 * ((nabla_obs2/(scale*scale)) + lambda * (nabla_smooth2)));
    // std::cout << "xi2 "<<(xi2).format(CleanFmt) << std::endl; //(Ainv2*nabla_obs2)/1000)
         //exit (EXIT_FAILURE);
  #else
  Vector dxi (Ainv * ( nabla_obs + lambda *nabla_smooth)); //
  //std::cout << "dxi "<< (dxi/eta).format(CleanFmt) << std::endl;
  //std::cout << "dxi "<< dxi.format(CleanFmt) << std::endl;
  xi -= dxi / eta;
  #endif
     auto finish = std::chrono::high_resolution_clock::now();
        elapsed+= finish - start;
        countvar++;
  //xi2 -= ((dxi2.cast<int>())/1000.0)/eta2;
  
  // end of "the" CHOMP iteration
  //////////////////////////////////////////////////
  
  update_robots ();

  }
  else{
    std::cout << "Elapsed time: " << elapsed.count()/countvar << " s\n";
    elapsed = std::chrono::seconds { 0 };
    countvar = 0;
  }
  // std::cout << "Elapsed time: " << elapsed.count()/4 << " s\n";
  //   elapsed = std::chrono::seconds { 0 };
}


static void cb_draw ()
{
  //////////////////////////////////////////////////
  // set bounds
  
  Vector bmin (qs);
  Vector bmax (qs);
  for (size_t ii (0); ii < 2; ++ii) {
    if (qe[ii] < bmin[ii]) {
      bmin[ii] = qe[ii];
    }
    if (qe[ii] > bmax[ii]) {
      bmax[ii] = qe[ii];
    }
    for (size_t jj (0); jj < nq; ++jj) {
      if (xi[ii + cdim * jj] < bmin[ii]) {
	bmin[ii] = xi[ii + cdim * jj];
      }
      if (xi[ii + cdim * jj] > bmax[ii]) {
	bmax[ii] = xi[ii + cdim * jj];
      }
    }
  }
  
  gfx::set_view (bmin[0] - 2.0, bmin[1] - 2.0, bmax[0] + 2.0, bmax[1] + 2.0);
  
  //////////////////////////////////////////////////
  // robots
  
  rstart.draw();
  for (size_t ii (0); ii < robots.size(); ++ii) {
    robots[ii].draw();
  }
  rend.draw();
  
  //////////////////////////////////////////////////
  // trj
  
  gfx::set_pen (1.0, 0.2, 0.2, 0.2, 1.0);
  gfx::draw_line (qs[0], qs[1], xi[0], xi[1]);
  for (size_t ii (1); ii < nq; ++ii) {
    gfx::draw_line (xi[(ii-1) * cdim], xi[(ii-1) * cdim + 1], xi[ii * cdim], xi[ii * cdim + 1]);
  }
  gfx::draw_line (xi[(nq-1) * cdim], xi[(nq-1) * cdim + 1], qe[0], qe[1]);
  
  gfx::set_pen (5.0, 0.8, 0.2, 0.2, 1.0);
  gfx::draw_point (qs[0], qs[1]);
  gfx::set_pen (5.0, 0.5, 0.5, 0.5, 1.0);
  for (size_t ii (0); ii < nq; ++ii) {
    gfx::draw_point (xi[ii * cdim], xi[ii * cdim + 1]);
  }
  gfx::set_pen (5.0, 0.2, 0.8, 0.2, 1.0);
  gfx::draw_point (qe[0], qe[1]);
  
  //////////////////////////////////////////////////
  // handles
  
  // for (handle_s ** hh (handle); *hh != 0; ++hh) {
  //   gfx::set_pen (1.0, (*hh)->red_, (*hh)->green_, (*hh)->blue_, (*hh)->alpha_);
  //   gfx::fill_arc ((*hh)->point_[0], (*hh)->point_[1], (*hh)->radius_, 0.0, 2.0 * M_PI);
  // }
    for (int ii = 0; ii < obs.cols(); ++ii) {
    gfx::set_pen(1.0, 0.0, 0.0, 1.0, 0.2);
    gfx::fill_arc(obs(0, ii), obs(1, ii), obs(2, ii), 0.0, 2.0 * M_PI);
  }
}
void add_obs(double px, double py, double radius)
{
  //conservativeResize is used. It's a costy operation, but hopefully will not
  // be done too often.
  obs.conservativeResize(obs_dim, obs.cols() + 1);
  obs.block(0, obs.cols() - 1, obs_dim, 1) << px, py, radius;
}

static void cb_mouse (double px, double py, int flags)
{
  if ((flags & gfx::MOUSE_RELEASE) && (flags & gfx::MOUSE_B3)) {
    //add new obstacle at that location
    add_obs(px, py, 2.0);
  } else if (flags & gfx::MOUSE_PRESS) {
    for (int ii = 0; ii < obs.cols(); ++ii) {
      Vector offset(obs.block(0, ii, 2, 1));
      offset[0] -= px;
      offset[1] -= py;
      if (offset.norm() <= obs(2, ii)) { //radius
        grab_offset = offset;
        grabbed = ii;
        state = RUN;
        break;
      }
    }
  } else if (flags & gfx::MOUSE_DRAG) {
    if (-1 != grabbed) {
      obs(0, grabbed) = px + grab_offset(0);
      obs(1, grabbed) = py + grab_offset(1);
    }
  } else if (flags & gfx::MOUSE_RELEASE) {
    grabbed = -1;
    //state = PAUSE;
  }
}
// {
//   if ((flags & gfx::MOUSE_RELEASE) && (flags & gfx::MOUSE_B3)) {
//     //add new obstacle at that location
//     add_obs(px, py, 2.0);
//   } else if (flags & gfx::MOUSE_PRESS) {
//     for (handle_s ** hh (handle); *hh != 0; ++hh) {
//       Vector offset ((*hh)->point_);
//       offset[0] -= px;
//       offset[1] -= py;
//       if (offset.norm() <= (*hh)->radius_) {
//     	grab_offset = offset;
//     	grabbed = *hh;
//     	break;
//       }
//     }
//   }
//   else if (flags & gfx::MOUSE_DRAG) {
//     if (0 != grabbed) {
//       grabbed->point_[0] = px;
//       grabbed->point_[1] = py;
//       grabbed->point_ += grab_offset;
//     }
//   }
//   else if (flags & gfx::MOUSE_RELEASE) {
//     grabbed = 0;
//   }
// }


int main()
{
   // struct timeval tt;
   // gettimeofday (&tt, NULL);
   // srand (tt.tv_usec);

       
        
            //    int fd;
            //    fd = open("/dev/ttyACM2",O_RDWR | O_NONBLOCK);//O_NOCTTY );  //

            //       struct pollfd fds[1];
            //     fds[0].fd = fd;
            //     fds[0].events = POLLIN ;
                

            //    if(fd == -1)           /* Error Checking */
            //                  printf("\n  Error! in Opening ttyACM1  ");
            //           else
            //                  printf("\n  ttyACM1 Opened Successfully ");

            //       struct termios SerialPortSettings;  /* Create the structure                          */

            //     tcgetattr(fd, &SerialPortSettings); /* Get the current attributes of the Serial port */

            //     /* Setting the Baud rate */ //B230400
            //     cfsetispeed(&SerialPortSettings,B230400); /* Set Read  Speed as 9600                       */
            //     cfsetospeed(&SerialPortSettings,B230400); // Set Write Speed as 9600
                
            //     SerialPortSettings.c_cflag &= ~PARENB;   /* Disables the Parity Enable bit(PARENB),So No Parity   */
            //     SerialPortSettings.c_cflag &= ~CSTOPB;   /* CSTOPB = 2 Stop bits,here it is cleared so 1 Stop bit */
            //     SerialPortSettings.c_cflag &= ~CSIZE;  /* Clears the mask for setting the data size             */
            //     SerialPortSettings.c_cflag |=  CS8;      /* Set the data bits = 8                                 */
            //     SerialPortSettings.c_cflag &= ~CRTSCTS;       /* No Hardware flow Control                         */
            //     SerialPortSettings.c_cflag |= CREAD | CLOCAL; /* Enable receiver,Ignore Modem Control lines       */ 
            //     SerialPortSettings.c_iflag &= ~(IXON | IXOFF | IXANY);          /* Disable XON/XOFF flow control both i/p and o/p */
            //     SerialPortSettings.c_iflag &= ~(ICANON | ECHO | ECHOE | ISIG);  /* Non Cannonical mode                            */
            //     SerialPortSettings.c_oflag &= ~OPOST;/*No Output Processing*/
            //     /* Setting Time outs */
            //     SerialPortSettings.c_cc[VMIN] = 8; /* Read at least 10 characters */
            //     SerialPortSettings.c_cc[VTIME] = 0; /* Wait indefinetly   */


            //     if((tcsetattr(fd,TCSANOW,&SerialPortSettings)) != 0) /* Set the attributes to the termios structure*/
            //         printf("\n  ERROR ! in Setting attributes");
            //     else
            //         printf("\n  BaudRate = 115200 \n  StopBits = 1 \n  Parity   = none");
                  
            //           /*------------------------------- Read data from serial port -----------------------------*/
            //     char write_buffer[] = "h";  /* Buffer containing characters to write into port       */ 
            //     int  bytes_written  = 0;    /* Value for storing the number of bytes written to the port */ 
            //     tcflush(fd, TCIFLUSH);    //Discards old data in the rx buffer            

            //     char read_buffer[20];    //Buffer to store the data received              
            //     int  bytes_read = 0;    /* Number of bytes read by the read() system call */
            //     int x = 10000000;
            //     int counter=0;
            //     int n = -1234000;//2147480000;//JJ2(0,3);
            //     int number2 = n;
            //     char numberStr[4];
            //     char numberStr2[5];
            //     memcpy(numberStr, &number2, 4);
            //     numberStr2[0] = 'h';
            //     numberStr2[1] = numberStr[0];
            //     numberStr2[2] = numberStr[1];
            //     numberStr2[3] = numberStr[2];
            //     numberStr2[4] = numberStr[3];
            //     //printf("%hhX %hhX %hhX %hhX\n", numberStr[0] , numberStr[1], numberStr[2], numberStr[3]);
                
            //     while(true)//counter <10)
            //     {
            //       counter+=1;
            //       n+=1;
            //       number2 = n;
            //       printf("number: %d \n", number2);

            //       memcpy(numberStr, &number2, 4);
            //       numberStr2[0] = 'h';
            //       numberStr2[1] = numberStr[0];
            //       numberStr2[2] = numberStr[1];
            //       numberStr2[3] = numberStr[2];
            //       numberStr2[4] = numberStr[3];
            //           //printf("\n  %d i", x); 
                                   

            //      auto start = std::chrono::high_resolution_clock::now();
            //       int pollrc = poll( fds, 1,10 );
            //      //auto temp = ("h"+std::to_string(x)+"g").c_str();
            //      //std::cout << "temp:  " << temp << std::endl;
            //     // std::string strtosend = "h" + std::string(numberStr);
            //      bytes_written = write(fd,  numberStr2 ,sizeof(numberStr2)); //strtosend.c_str()
            //                  printf("%hhX %hhX %hhX %hhX\n", numberStr[0] , numberStr[1], numberStr[2], numberStr[3]);

                 
            //     //bytes_written = write(fd, "a",1); 
            //     //for(int i =0; i<100; i++)
            //     //{
                 
            //     //"habhabhabhabhabhabhabhabhabhabhabhabhabhabhabhabhabhabhabhabhabhabhabhabhabhabhabhabhabhabhabhabhab"
            //     //,90);
            //     //"hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh"
            //     //, 96*8);//fd,write_buffer,sizeof(write_buffer));
            //     //}
            //     if( fds[0].revents & POLLIN )
            //     {
            //         //char buff[1024];
            //         //ssize_t rc = read(serial_fd, buff, sizeof(buff) );
            //          printf("\n  %d Bytes written to ttyUSB0", bytes_written);  
            //         printf("\n\n  Bytes Rxed :%d \n", bytes_read); /* Print the number of bytes read */
            //         bytes_read = read(fd,&read_buffer,20); /* Read the data                   */
            //         if ( bytes_read <= 0 )
            //           {
            //           cout << "Error " << errno << " opening " << "/dev/ttyUSB0" << ": " << strerror (errno) << endl;
            //           }
            //         if (bytes_read > 0)
            //         {
            //             /* You've got rc characters. do something with buff */
            //           for(int i=0;i<bytes_read;i++){  /*printing only the received characters*/
            //               printf("%c",read_buffer[i]);

            //             }
            //              printf("\n +----------------------------------+\n\n\n");
                       
            //           auto finish = std::chrono::high_resolution_clock::now();
            //           elapsed = finish - start;
            //           std::cout << "time:  " << elapsed.count() << endl;
            //           elapsed = std::chrono::seconds { 0 };
            //           printf("\n +----------------------------------+\n\n\n");
            //         }
            //     }

            //     //x = x +1;
            // }

        //     close(fd); /* Close the serial port */                                  
   // int x =10;
   // int y =10;
   // int A[x][y];
   // int B[x][y];
   // int C[x][y];
   // int i1, i2;
   // cout << endl;
   // cout << "---------------";

   // for (i1 = 0; i1 < x; i1++) {
   //  cout << endl;;
   //  for (i2 = 0; i2 < y; i2++) {
   //   A[i1][i2] = rand()% 100 + 1;
   //   cout << A[i1][i2] << "\t";
   //  }
   // }
   // cout << endl;
   // cout << "---------------";
   //  for (i1 = 0; i1 < x; i1++) {
   //  cout << endl;;
   //  for (i2 = 0; i2 < y; i2++) {
   //   B[i1][i2] = rand()% 100 + 1;
   //   cout << B[i1][i2] << "\t";
   //  }
   // }


  // std::cout << "gothere4" << std::endl;
 
  //     Eigen::MatrixXd a = Eigen::MatrixXd::Random(10, 10);
  //     Eigen::MatrixXd b = Eigen::MatrixXd::Random(10, 10);
  //      Eigen::MatrixXd c;
  //   auto start = std::chrono::high_resolution_clock::now();

  //   for(int x = 0; x <10000; ++x){
      
  //     c = a * b;
      
  //   }


  // for(int p = 0; p <10000; ++p){      
  //   for (int r = 0; r < x; r++) {
  //   for (int c = 0; c < y; c++) {
  //       for (int in = 0; in < x; in++) {
  //           C[r][c] += A[r][in] * B[in][c];
  //       }
  //       //cout << C[r][c] << "  ";
  //   }
  //   //cout << "\n";
  //   }
  // }

    // auto finish = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed = finish - start;
    // std::cout << "Elapsed time: " << elapsed.count() << " s\n";
  
  add_obs(3.0, 0.0, 2.0);
  add_obs(0.0, 3.0, 2.0);
  init_chomp();
  update_robots();  
  state = PAUSE;
  
  gfx::add_button ("jumble", cb_jumble);
  gfx::add_button ("step", cb_step);
  gfx::add_button ("run", cb_run);
  gfx::main ("chomp", cb_idle, cb_draw, cb_mouse);


}
