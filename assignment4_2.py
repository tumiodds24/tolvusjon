import cv2
import numpy as np

def compute_intersection(line1, line2):
    """Finds the intersection of two lines given in Hough space."""
    r1, theta1 = line1
    r2, theta2 = line2
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([r1, r2])
    x0, y0 = np.linalg.inv(A).dot(b)
    return int(np.round(x0)), int(np.round(y0))

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

        if lines is not None:
            lines = [l[0] for l in lines]  # Convert to a flat list of lines
            intersections = []
            for i in range(len(lines)):
                for j in range(i+1, len(lines)):
                    try:
                        intersect = compute_intersection(lines[i], lines[j])
                        intersections.append(intersect)
                    except np.linalg.LinAlgError:
                        # If the matrix is singular, skip these lines
                        continue

            if len(intersections) >= 4:
                # Assume the first four intersections are the corners
                corners = np.array(intersections[:4], dtype="float32")
                # Compute the max width and height for the new image
                width, height = 300, 200  # Arbitrary size
                destination_corners = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype='float32')
                M = cv2.getPerspectiveTransform(corners, destination_corners)
                dst = cv2.warpPerspective(frame, M, (width, height))
                cv2.imshow("Rectified Image", dst)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
