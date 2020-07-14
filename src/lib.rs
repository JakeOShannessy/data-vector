use chrono::prelude::*;
use serde::{Deserialize, Serialize};
use Debug;

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq, PartialOrd)]
pub struct DataVector<X, Y> {
    pub name: String,
    pub x_units: String,
    pub x_name: String,
    pub y_units: String,
    pub y_name: String,
    /// The values associated with this vector.
    values: Vec<Point<X, Y>>,
}

impl<X:PartialOrd,Y:PartialOrd> DataVector<X,Y> {
    pub fn new(name: String, x_units: String, x_name: String, y_units: String, y_name: String, mut values: Vec<Point<X, Y>>) -> Self {
        values.sort_unstable();
        Self {
            name,
            x_units,
            x_name,
            y_units,
            y_name,
            values,
        }
    }
    pub fn insert(&mut self, value: Point<X,Y>) {
        if let Some(i) = self.values.iter().position(|p| p.x > value.x) {
            self.values.insert(i,value);
        } else {
            self.values.push(value);
        }
    }
}
// We need an interpolatable trait. This includes this like dates.
pub trait Interpolate<X: PartialOrd> {
    fn interpolate(x: X, x1: X, x2: X, y1: Self, y2: Self) -> Self;
}

impl<X: Copy + PartialOrd + core::ops::Sub> Interpolate<X> for f64
where
    <X as std::ops::Sub>::Output: core::ops::Div<<X as std::ops::Sub>::Output, Output = f64>,
{
    fn interpolate(x: X, x1: X, x2: X, y1: Self, y2: Self) -> Self {
        ((x - x1) / (x2 - x1)) * (y2 - y1) + y1
    }
}

impl<Tz: TimeZone> Interpolate<f64> for chrono::DateTime<Tz> {
    fn interpolate(x: f64, x1: f64, x2: f64, y1: Self, y2: Self) -> Self {
        let basis = 1000_f64;
        let y_diff = y2 - y1.clone();
        y1 + (y_diff * (((x - x1) * basis) as i32) / (((x2 - x1) * basis) as i32)) / (basis as i32)
    }
}

// pub

impl<X, Y> DataVector<X, Y> {
    /// A getter method for the values. There is no field access as we don't
    /// want to allow arbitrary changing that might result in unordered data.
    pub fn values(&self) -> &Vec<Point<X, Y>> {
        &self.values
    }
}

// TODO: can just be ref
impl<X: Copy, Y: Copy> DataVector<X, Y> {
    pub fn iter(&self) -> impl Iterator<Item = (X, Y)> + '_ {
        self.values.iter().map(|p| (p.x, p.y))
    }
}

impl<
        X: Copy
            + Clone
            + PartialOrd
            + std::ops::Sub<Output = X>
            + std::ops::AddAssign
            + std::ops::Add<Output = X>
            + Zero,
        Y: Clone + PartialOrd + Interpolate<X>,
    > DataVector<X, Y>
{
    /// Resample to a fixed number of points along the x-axis
    pub fn resample_delta(&self, delta: X) -> Self {
        let mut new_values = Vec::new();
        let x_start = self.values.first().unwrap().x;
        let x_end = self.values.last().unwrap().x;
        let mut x_diff = X::zero();
        let x_step = delta;
        while x_diff < (x_end - x_start) {
            let x = x_start + x_diff;
            let y = self.interpolate(x).unwrap();
            new_values.push(Point::new(x, y));
            x_diff += x_step;
        }
        new_values.push(Point::new(x_end, self.interpolate(x_end).unwrap()));
        Self {
            name: self.name.clone(),
            x_units: self.x_units.clone(),
            x_name: self.x_name.clone(),
            y_units: self.y_units.clone(),
            y_name: self.y_name.clone(),
            values: new_values,
        }
    }
}


impl<
        X: Copy
            + Clone
            + PartialOrd
            + std::ops::Sub<Output = X>
            + std::ops::AddAssign
            + std::ops::Add<Output = X>
            + Zero,
        Y: Clone + PartialOrd + Interpolate<X>,
    > DataVector<X, Y>
{
    /// Smooth using a Savitzky–Golay filter
    pub fn smooth(&self, delta: X) -> Self {
        let mut new_values = Vec::new();
        let x_start = self.values.first().unwrap().x;
        let x_end = self.values.last().unwrap().x;
        let mut x_diff = X::zero();
        let x_step = delta;
        while x_diff < (x_end - x_start) {
            let x = x_start + x_diff;
            let y = self.interpolate(x).unwrap();
            new_values.push(Point::new(x, y));
            x_diff += x_step;
        }
        new_values.push(Point::new(x_end, self.interpolate(x_end).unwrap()));
        Self {
            name: self.name.clone(),
            x_units: self.x_units.clone(),
            x_name: self.x_name.clone(),
            y_units: self.y_units.clone(),
            y_name: self.y_name.clone(),
            values: new_values,
        }
    }
}


impl<
        X: Copy
            + Clone
            + PartialOrd
            + std::ops::Sub<Output = X>
            + std::ops::AddAssign
            + std::ops::Add<Output = X>
            + Zero
            + std::ops::Div<Output = X>,
        Y: Clone + PartialOrd + Interpolate<X>,
    > DataVector<X, Y>
{
    /// Resample to a fixed number of points along the x-axis
    pub fn resample_n<N: Into<X>>(&self, n: N) -> Self {
        let mut new_values = Vec::new();
        let x_start = self.values.first().unwrap().x;
        let x_end = self.values.last().unwrap().x;
        let mut x_diff = X::zero();
        let nx: X = n.into();
        let x_step = (x_end - x_start) / nx;
        while x_diff < (x_end - x_start) {
            let x = x_start + x_diff;
            let y = self.interpolate(x).unwrap();
            new_values.push(Point::new(x, y));
            x_diff += x_step;
        }
        new_values.push(Point::new(x_end, self.interpolate(x_end).unwrap()));
        Self {
            name: self.name.clone(),
            x_units: self.x_units.clone(),
            x_name: self.x_name.clone(),
            y_units: self.y_units.clone(),
            y_name: self.y_name.clone(),
            values: new_values,
        }
    }
}

impl<X: Copy + Clone + PartialOrd, Y: Clone + PartialOrd + Interpolate<X>> DataVector<X, Y> {
    pub fn combined_iter<'a>(&'a self, other: &'a DataVector<X, Y>) -> CombinedDVIter<X, Y> {
        CombinedDVIter {
            first_values: &self.values,
            second_values: &other.values,
            next_first_i: 0,
            next_second_i: 0,
        }
    }

    /// Resample self onto another vector. That is, interpolate to create a new
    /// vector with the same x-axis as ['other']. Actually, we need all the
    /// points of both to preserve accuracy.
    pub fn resample_max(&self, other: &DataVector<X, Y>, name: String) -> Self {
        let mut new_values = Vec::new();
        let value_iter = self.combined_iter(other);
        for value in value_iter {
            let point = match value {
                WhichVector::Both(p1, p2) => Point {
                    x: p1.x,
                    y: max_or_first(p1.y, p2.y),
                },
                WhichVector::First(p) => {
                    let y = match other.interpolate(p.x) {
                        Some(second_y) => max_or_first(p.y, second_y),
                        None => p.y,
                    };
                    Point { x: p.x, y }
                }
                WhichVector::Second(p) => {
                    let y = match self.interpolate(p.x) {
                        Some(first_y) => max_or_first(p.y, first_y),
                        None => p.y,
                    };
                    Point { x: p.x, y }
                }
            };
            new_values.push(point);
        }
        Self {
            name,
            x_units: self.x_units.clone(),
            x_name: self.x_name.clone(),
            y_units: self.y_units.clone(),
            y_name: self.y_name.clone(),
            values: new_values,
        }
    }
}

impl<X: Copy + PartialOrd + Clone, Y: Clone + PartialOrd + Interpolate<X>> DataVector<X, Y> {
    /// Truncate the vector at the given x-value, but re-interpolate to get the
    /// final value at the truncation point. Return None if no truncation was
    /// necessary. Interpolation is only applied if truncation occurs. TODO:
    /// This assumes that the vector is sorted, which we have not yet
    /// guaranteed.
    pub fn clip(&mut self, x: X) -> Option<()> {
        let y = self.interpolate(x).unwrap();
        // The first index above x.
        let x_index_above = self.values.iter().position(|p| p.x > x)?;
        self.values.truncate(x_index_above);
        self.values.push(Point::new(x, y));
        Some(())
    }
}

impl<X: PartialOrd, Y: PartialOrd> DataVector<X, Y> {
    pub fn sort(&mut self) {
        self.values.sort();
    }
}

impl<X: PartialOrd + Copy, Y: Copy + PartialOrd> DataVector<X, Y> {
    /// A getter method for the values. There is no field access as we don't
    /// want to allow arbitrary changing that might result in unordered data.
    pub fn bounds(&self) -> Option<(Point<X, Y>, Point<X, Y>)> {
        let mut values = self.values().iter();
        if let Some(first_value) = values.next() {
            let mut x_max = first_value.x;
            let mut x_min = first_value.x;
            let mut y_max = first_value.y;
            let mut y_min = first_value.y;
            for value in values {
                if value.x > x_max {
                    x_max = value.x;
                }
                if value.y > y_max {
                    y_max = value.y;
                }
                if value.x < x_min {
                    x_min = value.x;
                }
                if value.y < y_min {
                    y_min = value.y;
                }
            }
            Some((Point::new(x_min, y_min), Point::new(x_max, y_max)))
        } else {
            None
        }
    }
}

impl<
        X: Copy + Clone + PartialOrd,
        Y: Interpolate<X>
            + PartialOrd
            + Zero
            + One
            + Clone
            + core::ops::Add<Y, Output = Y>
            + core::ops::Div<Y, Output = Y>,
    > DataVector<X, Y>
{
    /// Resample as an average of two vectors.
    pub fn resample_avg(&self, other: &DataVector<X, Y>, name: String) -> Self {
        let mut new_values = Vec::new();
        let value_iter = self.combined_iter(other);
        for value in value_iter {
            let point = match value {
                WhichVector::Both(p1, p2) => Point {
                    x: p1.x,
                    y: (p1.y + p2.y) / (Y::one() + Y::one()),
                },
                WhichVector::First(p) => {
                    let y = other.interpolate(p.x).unwrap_or(Y::zero());
                    Point {
                        x: p.x,
                        y: (p.y + y) / (Y::one() + Y::one()),
                    }
                }
                WhichVector::Second(p) => {
                    let y = self.interpolate(p.x).unwrap_or(Y::zero());
                    Point {
                        x: p.x,
                        y: (p.y + y) / (Y::one() + Y::one()),
                    }
                }
            };
            new_values.push(point);
        }
        Self {
            name,
            x_units: self.x_units.clone(),
            x_name: self.x_name.clone(),
            y_units: self.y_units.clone(),
            y_name: self.y_name.clone(),
            values: new_values,
        }
    }
}

impl<X: Copy + Clone + PartialOrd, Y: Clone + PartialOrd + Interpolate<X>> DataVector<X, Y> {
    pub fn interpolate(&self, x: X) -> Option<Y> {
        if self.values.len() == 0 {
            return None;
        }
        // We assume that the values are properly sorted on the x-axis.
        for i in 0..self.values.len() {
            let this_point = self.values[i].clone();
            if x < this_point.x {
                return None;
            }
            if let Some(next_point) = self.values.get(i + 1) {
                if x > next_point.x {
                    continue;
                } else {
                    let x1 = this_point.x;
                    let x2 = next_point.x;
                    let y1 = this_point.y;
                    let y2 = next_point.y.clone();
                    // Value is between this_point and next_point.
                    return Some(Y::interpolate(x, x1, x2, y1, y2));
                }
            } else {
                return None;
            }
        }
        None
    }
}

pub trait Zero {
    fn zero() -> Self;
}

impl Zero for std::time::Duration {
    fn zero() -> Self {
        std::time::Duration::from_secs(0)
    }
}

impl Zero for f64 {
    fn zero() -> Self {
        0.0
    }
}

pub trait One {
    fn one() -> Self;
}

impl One for f64 {
    fn one() -> Self {
        1.0
    }
}

impl<X: Copy + PartialOrd + Clone, Y> DataVector<X, Y>
where
    Y: Clone + Zero + PartialOrd + Interpolate<X> + core::ops::Add<Y, Output = Y> + core::ops::Sub,
{
    /// Resample self onto another vector. That is, interpolate to create a new
    /// vector with the same x-axis as ['other']. Actually, we need all the
    /// points of both to preserve accuracy.
    pub fn resample_add(&self, other: &DataVector<X, Y>, name: String) -> Self {
        let mut new_values = Vec::new();
        let value_iter = self.combined_iter(other);
        for value in value_iter {
            let point: Point<X, Y> = match value {
                WhichVector::Both(p1, p2) => Point {
                    x: p1.x,
                    y: p1.y + p2.y,
                },
                WhichVector::First(p) => {
                    let y = other.interpolate(p.x).unwrap_or(Y::zero());
                    Point { x: p.x, y: p.y + y }
                }
                WhichVector::Second(p) => {
                    let y = self.interpolate(p.x).unwrap_or(Y::zero());
                    Point { x: p.x, y: p.y + y }
                }
            };
            new_values.push(point);
        }
        Self {
            name,
            x_units: self.x_units.clone(),
            x_name: self.x_name.clone(),
            y_units: self.y_units.clone(),
            y_name: self.y_name.clone(),
            values: new_values,
        }
    }
}

impl<X: Copy, Y> DataVector<X, Y> {
    // TODO: switch to AddAssign
    pub fn x_offset<T: core::ops::Add<X, Output = X> + Copy>(&mut self, offset: T) {
        for point in self.values.iter_mut() {
            point.x = offset + point.x;
        }
    }
}

impl<X: Copy, Y: Copy> DataVector<X, Y> {
    // TODO: switch to AddAssign
    pub fn y_offset<T: core::ops::Add<Y, Output = Y> + Copy>(&mut self, offset: T) {
        for point in self.values.iter_mut() {
            point.y = offset + point.y;
        }
    }
}

fn max_or_first<T: PartialOrd>(a: T, b: T) -> T {
    if b < a {
        a
    } else {
        b
    }
}

pub struct CombinedDVIter<'a, X, Y> {
    first_values: &'a Vec<Point<X, Y>>,
    second_values: &'a Vec<Point<X, Y>>,
    next_first_i: usize,
    next_second_i: usize,
}

impl<'a, X: Clone + PartialEq + PartialOrd, Y: Clone> Iterator for CombinedDVIter<'a, X, Y> {
    type Item = WhichVector<X, Y>;

    fn next(&mut self) -> Option<Self::Item> {
        // See what points are next.
        let next_first_point_opt = self.first_values.get(self.next_first_i);
        let next_second_point_opt = self.second_values.get(self.next_second_i);
        match (next_first_point_opt, next_second_point_opt) {
            (Some(first_point), None) => {
                self.next_first_i += 1;
                Some(WhichVector::First((*first_point).clone()))
            }
            (None, Some(second_point)) => {
                self.next_second_i += 1;
                Some(WhichVector::Second((*second_point).clone()))
            }
            (Some(first_point), Some(second_point)) => {
                if first_point.x < second_point.x {
                    self.next_first_i += 1;
                    Some(WhichVector::First((*first_point).clone()))
                } else if second_point.x < first_point.x {
                    self.next_second_i += 1;
                    Some(WhichVector::Second((*second_point).clone()))
                } else if first_point.x == second_point.x {
                    self.next_first_i += 1;
                    self.next_second_i += 1;
                    Some(WhichVector::Both(
                        (*first_point).clone(),
                        (*second_point).clone(),
                    ))
                } else {
                    panic!("invalid equality test");
                }
            }
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum WhichVector<X, Y> {
    /// We have a value from the first vector but not the second.
    First(Point<X, Y>),
    /// We have a value from the seconds vector but not the first.
    Second(Point<X, Y>),
    /// We have values from both vectors.
    Both(Point<X, Y>, Point<X, Y>),
}

#[derive(Copy, Clone, Debug, Deserialize, Serialize, PartialEq, PartialOrd)]
pub struct Point<X, Y> {
    pub x: X,
    pub y: Y,
}

impl<X, Y> Point<X, Y> {
    pub fn new(x: X, y: Y) -> Self {
        Self { x, y }
    }
}

use std::cmp::Ordering;
impl<X: PartialOrd, Y: PartialOrd> Ord for Point<X, Y> {
    // Assumes that NaN is less.
    fn cmp(&self, other: &Self) -> Ordering {
        if self.x > other.x {
            Ordering::Greater
        } else if self.x == other.x {
            Ordering::Equal
        } else {
            Ordering::Less
        }
    }
}

impl<X: PartialEq, Y: PartialEq> Eq for Point<X, Y> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolation() {
        let dv = DataVector {
            name: "Test DV".to_string(),
            x_units: "s".to_string(),
            x_name: "Time".to_string(),
            y_units: "kW".to_string(),
            y_name: "HRR".to_string(),
            values: vec![
                Point { x: 0_f64, y: 0_f64 },
                Point {
                    x: 100_f64,
                    y: 100_f64,
                },
            ],
        };
        assert_eq!(Some(50.0), dv.interpolate(50.0));
        assert_eq!(Some(75.0), dv.interpolate(75.0));
    }

    #[test]
    fn resample_n() {
        let dv = DataVector {
            name: "Test DV".to_string(),
            x_units: "s".to_string(),
            x_name: "Time".to_string(),
            y_units: "kW".to_string(),
            y_name: "HRR".to_string(),
            values: vec![
                Point { x: 0_f64, y: 0_f64 },
                Point {
                    x: 100_f64,
                    y: 100_f64,
                },
                Point {
                    x: 200_f64,
                    y: 200_f64,
                },
                Point {
                    x: 300_f64,
                    y: 300_f64,
                },
            ],
        };
        let dv2 = dv.resample_n(2.0_f64);
        assert_eq!(
            dv2.values,
            vec![
                Point { x: 0_f64, y: 0_f64 },
                Point {
                    x: 150_f64,
                    y: 150_f64,
                },
                Point {
                    x: 300_f64,
                    y: 300_f64,
                },
            ]
        )
        // assert_eq!(Some(50.0), dv.interpolate(50.0));
        // assert_eq!(Some(75.0), dv.interpolate(75.0));
    }

    #[test]
    fn resample_delta() {
        let dv = DataVector {
            name: "Test DV".to_string(),
            x_units: "s".to_string(),
            x_name: "Time".to_string(),
            y_units: "kW".to_string(),
            y_name: "HRR".to_string(),
            values: vec![
                Point { x: 0_f64, y: 0_f64 },
                Point {
                    x: 100_f64,
                    y: 100_f64,
                },
                Point {
                    x: 200_f64,
                    y: 200_f64,
                },
                Point {
                    x: 300_f64,
                    y: 300_f64,
                },
            ],
        };
        let dv2 = dv.resample_delta(300.0);
        assert_eq!(
            dv2.values,
            vec![
                Point { x: 0_f64, y: 0_f64 },
                Point {
                    x: 300_f64,
                    y: 300_f64,
                },
            ]
        )
        // assert_eq!(Some(50.0), dv.interpolate(50.0));
        // assert_eq!(Some(75.0), dv.interpolate(75.0));
    }

    #[test]
    fn iters() {
        let dv1 = DataVector {
            name: "Test DV".to_string(),
            x_units: "s".to_string(),
            x_name: "Time".to_string(),
            y_units: "kW".to_string(),
            y_name: "HRR".to_string(),
            values: vec![
                Point { x: 0_f64, y: 0_f64 },
                Point {
                    x: 100_f64,
                    y: 100_f64,
                },
            ],
        };
        let dv2 = DataVector {
            name: "Test DV".to_string(),
            x_units: "s".to_string(),
            x_name: "Time".to_string(),
            y_units: "kW".to_string(),
            y_name: "HRR".to_string(),
            values: vec![
                Point {
                    x: 10_f64,
                    y: 0_f64,
                },
                Point {
                    x: 70_f64,
                    y: 100_f64,
                },
            ],
        };
        let ci: Vec<WhichVector<f64, f64>> = dv1.combined_iter(&dv2).collect();
        eprintln!("{:?}", ci);
    }

    #[test]
    fn add_vectors() {
        let dv1 = DataVector {
            name: "Test DV".to_string(),
            x_units: "s".to_string(),
            x_name: "Time".to_string(),
            y_units: "kW".to_string(),
            y_name: "HRR".to_string(),
            values: vec![
                Point { x: 0_f64, y: 0_f64 },
                Point {
                    x: 100_f64,
                    y: 100_f64,
                },
            ],
        };
        let dv2 = DataVector {
            name: "Test DV".to_string(),
            x_units: "s".to_string(),
            x_name: "Time".to_string(),
            y_units: "kW".to_string(),
            y_name: "HRR".to_string(),
            values: vec![
                Point {
                    x: 10_f64,
                    y: 0_f64,
                },
                Point {
                    x: 70_f64,
                    y: 100_f64,
                },
            ],
        };
        let dv3 = DataVector {
            name: "Test DV".to_string(),
            x_units: "s".to_string(),
            x_name: "Time".to_string(),
            y_units: "kW".to_string(),
            y_name: "HRR".to_string(),
            values: vec![
                Point { x: 0_f64, y: 0_f64 },
                Point {
                    x: 10_f64,
                    y: 10_f64,
                },
                Point {
                    x: 70_f64,
                    y: 170_f64,
                },
                Point {
                    x: 100_f64,
                    y: 100_f64,
                },
            ],
        };
        let ci = dv1.resample_add(&dv2, "Test DV".to_string());
        eprintln!("{:?}", ci);
        assert_eq!(dv3, ci);
    }

    #[test]
    fn max_vectors() {
        let dv1 = DataVector {
            name: "Test DV".to_string(),
            x_units: "s".to_string(),
            x_name: "Time".to_string(),
            y_units: "kW".to_string(),
            y_name: "HRR".to_string(),
            values: vec![
                Point { x: 0_f64, y: 0_f64 },
                Point {
                    x: 100_f64,
                    y: 100_f64,
                },
            ],
        };
        let dv2 = DataVector {
            name: "Test DV".to_string(),
            x_units: "s".to_string(),
            x_name: "Time".to_string(),
            y_units: "kW".to_string(),
            y_name: "HRR".to_string(),
            values: vec![
                Point {
                    x: 10_f64,
                    y: 0_f64,
                },
                Point {
                    x: 70_f64,
                    y: 100_f64,
                },
            ],
        };
        let dv3 = DataVector {
            name: "Test DV".to_string(),
            x_units: "s".to_string(),
            x_name: "Time".to_string(),
            y_units: "kW".to_string(),
            y_name: "HRR".to_string(),
            values: vec![
                Point { x: 0_f64, y: 0_f64 },
                Point {
                    x: 10_f64,
                    y: 10_f64,
                },
                Point {
                    x: 70_f64,
                    y: 100_f64,
                },
                Point {
                    x: 100_f64,
                    y: 100_f64,
                },
            ],
        };
        let ci = dv1.resample_max(&dv2, "Test DV".to_string());
        eprintln!("{:?}", ci);
        assert_eq!(dv3, ci);
    }
}
