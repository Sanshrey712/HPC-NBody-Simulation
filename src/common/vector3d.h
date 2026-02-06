#ifndef VECTOR3D_H
#define VECTOR3D_H

#include <cmath>
#include <iostream>

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

struct Vector3D {
    double x, y, z;

    CUDA_HOSTDEV Vector3D() : x(0.0), y(0.0), z(0.0) {}
    CUDA_HOSTDEV Vector3D(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    // Vector addition
    CUDA_HOSTDEV Vector3D operator+(const Vector3D& v) const {
        return Vector3D(x + v.x, y + v.y, z + v.z);
    }

    // Vector subtraction
    CUDA_HOSTDEV Vector3D operator-(const Vector3D& v) const {
        return Vector3D(x - v.x, y - v.y, z - v.z);
    }

    // Scalar multiplication
    CUDA_HOSTDEV Vector3D operator*(double s) const {
        return Vector3D(x * s, y * s, z * s);
    }

    // Scalar division
    CUDA_HOSTDEV Vector3D operator/(double s) const {
        return Vector3D(x / s, y / s, z / s);
    }

    // Compound assignment operators
    CUDA_HOSTDEV Vector3D& operator+=(const Vector3D& v) {
        x += v.x; y += v.y; z += v.z;
        return *this;
    }

    CUDA_HOSTDEV Vector3D& operator-=(const Vector3D& v) {
        x -= v.x; y -= v.y; z -= v.z;
        return *this;
    }

    CUDA_HOSTDEV Vector3D& operator*=(double s) {
        x *= s; y *= s; z *= s;
        return *this;
    }

    CUDA_HOSTDEV Vector3D& operator/=(double s) {
        x /= s; y /= s; z /= s;
        return *this;
    }

    // Dot product
    CUDA_HOSTDEV double dot(const Vector3D& v) const {
        return x * v.x + y * v.y + z * v.z;
    }

    // Cross product
    CUDA_HOSTDEV Vector3D cross(const Vector3D& v) const {
        return Vector3D(
            y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x
        );
    }

    // Magnitude squared (avoids sqrt)
    CUDA_HOSTDEV double magnitude_sq() const {
        return x * x + y * y + z * z;
    }

    // Magnitude
    CUDA_HOSTDEV double magnitude() const {
        return sqrt(magnitude_sq());
    }

    // Normalized vector
    CUDA_HOSTDEV Vector3D normalized() const {
        double mag = magnitude();
        if (mag > 0.0) {
            return *this / mag;
        }
        return Vector3D();
    }

    // Distance to another vector
    CUDA_HOSTDEV double distance_to(const Vector3D& v) const {
        return (*this - v).magnitude();
    }

    // Distance squared (for efficiency)
    CUDA_HOSTDEV double distance_sq_to(const Vector3D& v) const {
        return (*this - v).magnitude_sq();
    }

    // Print (CPU only)
    friend std::ostream& operator<<(std::ostream& os, const Vector3D& v) {
        os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
        return os;
    }
};

// Scalar * Vector
CUDA_HOSTDEV inline Vector3D operator*(double s, const Vector3D& v) {
    return v * s;
}

#endif // VECTOR3D_H
