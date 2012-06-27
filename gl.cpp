/*
	(C) 2012 Jeff Chien
	
	OpenGL visualization.
 */

#include <cstdlib>
#include <GL/glut.h>

#include "gl.h"

static pdla::pdla_result_t displaying;
static float maxRad, maxRadPadding;
static int numPart, partIndex;

static void renderCircle(pdla::vec v)
{
	glBegin(GL_LINE_LOOP);
	{
		for(float theta = 0; theta < TAU; theta += 0.2)
			glVertex3f(v.x + cosf(theta), v.y + sinf(theta), 0);
	}
	glEnd();
}

static void setWindowTitle()
{
	char buf[128];
	sprintf(buf, "PDLA: %d/%d seeds, time step %d", partIndex, numPart, displaying.time[partIndex - 1]);
	glutSetWindowTitle(buf);
}

static void display()
{
	glClear(GL_COLOR_BUFFER_BIT);
    glLineWidth(2);
	glColor3f(1.0f, 1.0f, 1.0f);
    glBegin(GL_LINES);
	{
		glVertex2f(0, maxRadPadding);
		glVertex2f(0, -maxRadPadding);

		glVertex2f(maxRadPadding, 0);
		glVertex2f(-maxRadPadding, 0);
	}
    glEnd();
	
    glLineWidth(1);
	glColor3f(0, 0, 1.0f);
	for(int i = 0; i < partIndex - 1; i++)
		renderCircle(displaying.pos[i]);
	glColor3f(0, 1.0f, 0);
	renderCircle(displaying.pos[partIndex - 1]);

	glutSwapBuffers();
}

static void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei)w, (GLsizei)h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(-maxRadPadding, maxRadPadding, -maxRadPadding, maxRadPadding);
}

static void keyboard(unsigned char key, int x, int y)
{
	int redraw = 1;
	// Exit on escape, q
	switch(key)
	{
		case 0x1b:
		case 'q':
		case 'Q':
			exit(0);
			break;

		case 's':
		case 'S':
			partIndex = 1;
			break;

		case 'e':
		case 'E':
			partIndex = numPart;
			break;

		default:
			redraw = 0;
			break;
	}
	if(redraw)
	{
		setWindowTitle();
		glutPostRedisplay();
	}
}

static void special(int key, int x, int y)
{
	int redraw = 1;
	switch(key)
	{
		case GLUT_KEY_LEFT:
			partIndex--;
			break;

		case GLUT_KEY_RIGHT:
			partIndex++;
			break;

		case GLUT_KEY_UP:
			partIndex -= 5;
			break;

		case GLUT_KEY_DOWN:
			partIndex += 5;
			break;

		default:
			redraw = 0;
			break;
	}
	if(partIndex > numPart)
		partIndex = numPart;
	else if(partIndex < 1)
		partIndex = 1;
	if(redraw)
	{
		setWindowTitle();
		glutPostRedisplay();
	}
}

void pdla::render(int argc, char** argv, const pdla_result_t& p)
{
	displaying = p;
	maxRad = 0;
	numPart = partIndex = p.pos.size();
	for(int i = 0; i < numPart; i++)
		if(p.pos[i].len() > maxRad)
			maxRad = p.pos[i].len();
	maxRadPadding = maxRad * 1.2f;

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(600, 600);
	glutCreateWindow("");
	setWindowTitle();
	glClearColor(0, 0, 0, 1);

	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(special);
	glutMainLoop();
}
