#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "flyBall.h"

namespace py = pybind11;

static flyBall flyBall_Obj;

// 归一化函数
double normalize(double x, double x_min, double x_max){
  return (x - x_min) / (x_max - x_min);
}

void rt_OneStep(void);
void rt_OneStep(void)
{
  static boolean_T OverrunFlag{ false };

  if (OverrunFlag) {
    rtmSetErrorStatus(flyBall_Obj.getRTM(), "Overrun");
    return;
  }

  OverrunFlag = true;
  flyBall_Obj.step();
  OverrunFlag = false;
}

int_T main(int_T argc, const char *argv[])
{
  (void)(argc);
  (void)(argv);
  flyBall_Obj.initialize();
  // done = ((rtmGetErrorStatus(flyBall_Obj.getRTM()) != (nullptr)) ||
  //       rtmGetStopRequested(flyBall_Obj.getRTM()))
  while ((rtmGetErrorStatus(flyBall_Obj.getRTM()) == (nullptr)) &&
         !rtmGetStopRequested(flyBall_Obj.getRTM())) {
    rt_OneStep();
  }

  flyBall_Obj.terminate();
  return 0;
}

void reset(){
  flyBall_Obj.initialize();
}

void step(double control){
  flyBall_Obj.set_control(control);
  rt_OneStep();
}

// 获取真实状态量state=[t, y, vy, ay, target_t, target_y, target_vy]
py::array_t<double> get_real_state(){
  double state[7];
  state[0] = flyBall_Obj.get_t();
  state[1] = flyBall_Obj.get_y();
  state[2] = flyBall_Obj.get_vy();
  state[3] = flyBall_Obj.get_ay();
  state[4] = flyBall_Obj.get_target_t();
  state[5] = flyBall_Obj.get_target_y();
  state[6] = flyBall_Obj.get_target_vy();
  return py::array_t<double>(7, state);
}

// 获取归一化后的状态量state=[t, y, vy, ay, target_t, target_y, target_vy]
py::array_t<double> state(){
  double state[7];
  state[0] = normalize(flyBall_Obj.get_t(), 0, 10);
  state[1] = normalize(flyBall_Obj.get_y(), 0, 100);
  state[2] = normalize(flyBall_Obj.get_vy(), -50, 0);
  state[3] = normalize(flyBall_Obj.get_ay(), -20, 2);
  state[4] = normalize(flyBall_Obj.get_target_t(), 0, 10);
  state[5] = normalize(flyBall_Obj.get_target_y(), 0, 100);
  state[6] = normalize(flyBall_Obj.get_target_vy(), -50, 0);
  return py::array_t<double>(7, state);
}

// 获取reward
double reward(){
  return flyBall_Obj.get_reward();
}

// 获取是否结束
bool done(){
  return ((rtmGetErrorStatus(flyBall_Obj.getRTM()) != (nullptr)) ||
       rtmGetStopRequested(flyBall_Obj.getRTM()));
}

// 设置控制量
void set_control(double control){
  flyBall_Obj.set_control(control);
}

PYBIND11_MODULE(flyBall, m) {
  m.def("reset", &reset);
  m.def("step", &step);
  m.def("get_real_state", &get_real_state);
  m.def("state", &state);
  m.def("reward", &reward);
  m.def("done", &done);
  m.def("set_control", &set_control);
}


