#include "mainwindow.h"
#include "ui_mainwindow.h"


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    connect(ui->random, SIGNAL (clicked()),this, SLOT (random()));


    // ///////////////////////////////////////
    for(int i = 0; i<3; i++){
        for (int j=0;j<J;j++){
            Wij[i][j] = ((double)rand() / (double)(RAND_MAX)) * 2.0 - 1.0;
        }
    }
    for(int j = 0; j <J ; j++){
        for(int k = 0; k < K; k++){
            Wjk[j][k] = ((double)rand() / (double)(RAND_MAX)) * 2.0 - 1.0;
        }
    }

    for(int k = 0; k<K; k++)
    {
        for(int l = 0; l<2; l++)
        {
            Wkl[k][l] = ((double)rand() / (double)(RAND_MAX)) * 2.0 - 1.0;
        }
    }
    // /////////////////////////////////////////

    dataGen();
    training();
}


double MainWindow::sigmoid(double x) {
    return (1.0f / (1.0f + std::exp(-x)));
}


void MainWindow::mouseMoveEvent(QMouseEvent *e){
    double xx = e->pos().x() - points[0];
    double yy =  e->pos().y() - points[1];
    points2[0] = xx * cos(M_PI/2.0) - yy * sin(M_PI/2.0);
    points2[1] = xx * sin(M_PI/2.0) +yy * cos(M_PI/2.0);

    memset(a, 0, J);
    memset(y,0,J);
    memset(b,0,2);
    memset(z,0,2);
    //skalowanie danych
    double trainData2[3];
    trainData2[0] = points2[0] / 500.0 * 0.8 +0.1;
    trainData2[1] = points2[1] / 500.0 * 0.8 +0.1;
    trainData2[2] = 1;
    /* suma warstwy 1 - aj po sigmoidzie / delta - yj
     * output - bk po sigmoidzie / delta - zk
     */

    //z input do WU1
    for (int j = 0 ; j<J; j++)
    {
        for (int i = 0; i< 3; i++)
        {
            a[j] += trainData2[i] * Wij[i][j];
        }
        a[j] = sigmoid(a[j]);
    }


    //z WU1 do WU2
    for (int k = 0; k <K; k++)
    {
        for(int j = 0 ; j<J-1; j++)
        {
            b[k] += a[j] * Wjk[j][k];
        }
        b[k] += Wjk[J-1][k]; // bias

        b[k] = sigmoid(b[k]);

    }

    //z WU2 do output

    for (int l = 0; l <2; l++)
    {
        for(int k = 0 ; k<K-1; k++)
        {

            c[l] += b[k] * Wkl[k][l];
        }
        c[l] += Wkl[K-1][l]; // bias

        c[l] = sigmoid(c[l]);

    }

    alpha = (c[0]-0.1) * M_PI * 1.25 - M_PI/2;
    beta = (c[1] - 0.1) * M_PI * 1.25;
    update();

}

void MainWindow::paintEvent(QPaintEvent *e)
{
    QPainter painter;
    painter.begin(this);
    QPen pen(Qt::black);
    pen.setWidth(5);
    painter.setPen(pen);
    painter.drawLine(points[0], points[1], points[0]+ (int)(arm * cos(alpha)),points[1]+(int)(arm * sin(alpha)));
    painter.drawLine(points[0] + (int)(arm * cos(alpha)),points[1]+(int)(arm * sin(alpha)),points[0]+(int)(arm2*cos(alpha-beta))+(int)(arm*cos(alpha)), points[1]+(int)(arm2*sin(alpha-beta)+(int)(arm*sin(alpha))));
}


void MainWindow::dataGen()
{
    for(int i = 0; i< N; i++){
    double a = (double)rand() / (double)RAND_MAX * M_PI;
    double b = (double)rand() / (double)RAND_MAX * M_PI;
    trainData[i][0] = a;
    trainData[i][1] = b;
    trainData[i][2] = 1.0;
    trainData[i][3] = (int)(arm2*cos(a-b))+(int)(arm*cos(a));
    trainData[i][4] = (int)(arm2*sin(a-b)+(int)(arm*sin(a)));
    }

}


void MainWindow::training()
{
    double errorr=9000000000;
    double prevError =10000000000;
    while(errorr > 25.0f)
//    for(int www = 0 ; www < 500 ; www++)
    {
        prevError = errorr;

        errorr = 0;
        std::random_shuffle(std::begin(trainData), std::end(trainData));
            for(int x=0; x<N; x++){
                memset(a, 0, J);
                memset(y,0,J);
                memset(b,0,2);
                memset(z,0,2);
                //skalowanie danych
                double trainData2[3];
                double expectedOutput[2];
                trainData2[0] = trainData[x][3] / 500.0 * 0.8 +0.1;
                trainData2[1] = trainData[x][4] / 500.0 * 0.8 +0.1;
                trainData2[2] = 1;
                expectedOutput[0] = trainData[x][0] / (double)M_PI * 0.8 + 0.1;
                expectedOutput[1] = trainData[x][1] / (double)M_PI * 0.8 + 0.1;
                /* suma warstwy 1 - aj po sigmoidzie / delta - yj
                 * output - bk po sigmoidzie / delta - zk
                 */

                //W przód
                //z input do WU1
                for (int j = 0 ; j<J; j++)
                {
                    for (int i = 0; i< 3; i++)
                    {
                        a[j] += trainData2[i] * Wij[i][j];
//                        qDebug() << trainData2[i] << Wij[i][j];
                    }
//                    qDebug() << "xxx"<<a[j] << j;
                    a[j] = sigmoid(a[j]);
//                    qDebug() <<a[j];
                }


                //z WU1 do WU2
                for (int k = 0; k <K; k++)
                {
                    for(int j = 0 ; j<J-1; j++)
                    {
//                        qDebug() << a[j] << Wjk[j][k] <<j; // aj za duże dla j=0
                        b[k] += a[j] * Wjk[j][k];
                    }
                    b[k] += Wjk[J-1][k]; // bias
//                    qDebug() << b[k] << k;
                    b[k] = sigmoid(b[k]);
//                    qDebug() << b[k] << k;
                }

                //z WU2 do output

                for (int l = 0; l <2; l++)
                {
                    for(int k = 0 ; k<K-1; k++)
                    {

                        c[l] += b[k] * Wkl[k][l];
                    }
                    c[l] += Wkl[K-1][l]; // bias

                    c[l] = sigmoid(c[l]);

                }


                //cofanie
                for(int l = 0; l<2;l++)
                {
                    q[l]=(c[l] - expectedOutput[l])* c[l]*(1-c[l]) ;
                }

                for(int k= 0; k<K; k++)
                {
                    for(int l=0;l<2; l++)
                    {
                        z[k] += (q[l] * Wkl[k][l]);
                    }
                    z[k] *= b[k]*(1.0-b[k]);
                }

                for(int j = 0; j<J;j++)
                {
                    for(int k = 0; k<K;k++)
                    {
                        y[j] += (z[k] * Wjk[j][k]);
                    }
                    y[j] *= a[j]*(1.0-a[j]);
                }

                //zmiana wag

                for(int i = 0; i< 3; i++)
                    for(int j = 0; j< J; j++)
                        Wij[i][j] = Wij[i][j] - su * y[j] * trainData2[i];

                for(int j = 0 ; j < J; j++)
                    for(int k = 0; k<K;k++)
                        Wjk[j][k] = Wjk[j][k] - su * z[k] * a[j];

                for(int k = 0 ; k<K; k++)
                    for(int l = 0; l<2;l++)
                        Wkl[k][l] = Wkl[k][l] - su * q[l] * b[k];

                for(int l = 0; l<2;l++)
                {
                    errorr+= (c[l] - expectedOutput[l])*(c[l] - expectedOutput[l]);
                }
        }
        errorr/=2.0;
//        su = (errorr / 60.0) * (errorr / 60.0);
        qDebug()<<25/errorr * 100.0 <<"%";

    }
}

void MainWindow::random()
{
    int x = rand()%N;
    alpha = trainData[x][0];
    beta = trainData[x][1];

    update();
}

MainWindow::~MainWindow()
{
    delete ui;
}
