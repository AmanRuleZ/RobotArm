#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QtCore>
#include <QtGui>
#include <cmath>
#include <vector>
#include <iterator>
#include <iostream>

#define J 30 //ilość elementów w wartwie ukrytej 1
#define K 30 // ilość elementów w warstwie ukrytej 2
#define N 10000 //ilość danych treningowych
namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    int arm = 120;
    int arm2 = 80;
    double alpha = 0.0;
    double beta = 0.0;
    double su = 0.1;

    double a[J] = {};
    double y[J] = {};
    double b[K] = {};
    double z[K] = {};

    double c[2] = {};
    double q[2] = {};

    double Wij[3][J];
    double Wjk[J][K];
    double Wkl[K][2];

    int points2[2] = {0,0};
    int points[2] = {220,220};
    double trainData[N][5];
    double sigmoid(double x);
    void training();
    void mouseMoveEvent(QMouseEvent *e);
    void paintEvent(QPaintEvent *e);
    void dataGen();
public slots:
    void random();
private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
